"""Train scikit-learn income-bracket prediction models."""

from __future__ import annotations

import math

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from config import (
    BEST_MODEL_PATH,
    CHUNK_SIZE,
    CLASSIFICATION_REPORT_PATH,
    CLEAN_DATA_PATH,
    CONFUSION_MATRIX_PATH,
    FEATURE_IMPORTANCE_PATH,
    INCOME_BRACKETS,
    MAX_MODEL_ROWS,
    MAX_SCORED_EXPORT_ROWS,
    MODEL_COMPARISON_EXPORT_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_SCORED_PATH,
    RANDOM_STATE,
    ensure_directories,
)


NUMERIC_FEATURES = ["age", "famsize"]
FULL_CATEGORICAL_FEATURES = [
    "state_name",
    "age_group",
    "sex_label",
    "race_group",
    "marital_status",
    "citizenship",
    "education_group",
    "employment_status",
    "degree_field_group",
    "second_degree_field_group",
]
REDUCED_CATEGORICAL_FEATURES = [
    feature
    for feature in FULL_CATEGORICAL_FEATURES
    if feature not in {"sex_label", "race_group"}
]
TARGET = "income_bracket"


def load_modeling_sample(path=CLEAN_DATA_PATH, max_rows: int = MAX_MODEL_ROWS) -> pd.DataFrame:
    """Read a representative sample from the clean dataset."""
    print(f"Loading up to {max_rows:,} rows for modeling...")
    samples = []
    per_chunk = max(5_000, math.ceil(max_rows / 12))
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
        if chunk.empty:
            continue
        n = min(per_chunk, len(chunk))
        samples.append(chunk.sample(n=n, random_state=RANDOM_STATE))
    if not samples:
        raise ValueError("No rows available for modeling.")
    data = pd.concat(samples, ignore_index=True)
    if len(data) > max_rows:
        data = data.sample(n=max_rows, random_state=RANDOM_STATE).reset_index(drop=True)
    data = data[data[TARGET].isin(INCOME_BRACKETS)].copy()
    print(f"Modeling sample contains {len(data):,} rows.")
    return data


def make_one_hot_preprocessor(categorical_features: list[str]) -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=25)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", encoder),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def make_ordinal_preprocessor(categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def build_models(categorical_features: list[str]) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", make_one_hot_preprocessor(categorical_features)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=600,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", make_one_hot_preprocessor(categorical_features)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=160,
                        min_samples_leaf=10,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                        class_weight="balanced_subsample",
                    ),
                ),
            ]
        ),
        "Hist Gradient Boosting": Pipeline(
            steps=[
                ("preprocess", make_ordinal_preprocessor(categorical_features)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=160,
                        learning_rate=0.08,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def train_income_models() -> pd.DataFrame:
    """Train full and reduced models, save metrics, best model, and scored rows."""
    ensure_directories()
    data = load_modeling_sample()
    full_features = NUMERIC_FEATURES + FULL_CATEGORICAL_FEATURES
    reduced_features = NUMERIC_FEATURES + REDUCED_CATEGORICAL_FEATURES

    for feature in set(full_features + reduced_features):
        if feature not in data.columns:
            data[feature] = np.nan
    for feature in NUMERIC_FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
        if data[feature].isna().all():
            data[feature] = 0

    data = data.dropna(subset=[TARGET]).copy()
    train_df, test_df = train_test_split(
        data,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=data[TARGET],
    )

    results = []
    reports = []
    trained_models = {}
    experiment_specs = {
        "full": FULL_CATEGORICAL_FEATURES,
        "reduced_ethics": REDUCED_CATEGORICAL_FEATURES,
    }

    for experiment_name, categorical_features in experiment_specs.items():
        features = NUMERIC_FEATURES + categorical_features
        for model_name, pipeline in build_models(categorical_features).items():
            print(f"Training {experiment_name} {model_name}...")
            pipeline.fit(train_df[features], train_df[TARGET])
            predictions = pipeline.predict(test_df[features])
            accuracy = accuracy_score(test_df[TARGET], predictions)
            macro_f1 = f1_score(test_df[TARGET], predictions, average="macro")
            weighted_f1 = f1_score(test_df[TARGET], predictions, average="weighted")
            results.append(
                {
                    "experiment": experiment_name,
                    "model": model_name,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "weighted_f1": weighted_f1,
                    "training_rows": len(train_df),
                    "test_rows": len(test_df),
                }
            )
            report = classification_report(
                test_df[TARGET],
                predictions,
                labels=INCOME_BRACKETS,
                output_dict=True,
                zero_division=0,
            )
            reports.append(flatten_report(report, experiment_name, model_name))
            trained_models[(experiment_name, model_name)] = (pipeline, features, predictions)

    comparison = pd.DataFrame(results).sort_values(
        ["weighted_f1", "macro_f1", "accuracy"], ascending=False
    )
    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    comparison.to_csv(MODEL_COMPARISON_EXPORT_PATH, index=False)

    pd.concat(reports, ignore_index=True).to_csv(CLASSIFICATION_REPORT_PATH, index=False)

    best_key = (comparison.iloc[0]["experiment"], comparison.iloc[0]["model"])
    best_model, best_features, best_predictions = trained_models[best_key]
    joblib.dump({"model": best_model, "features": best_features, "target": TARGET}, BEST_MODEL_PATH)
    print(f"Best model: {best_key[0]} {best_key[1]}")

    save_confusion_matrix(test_df[TARGET], best_predictions, best_key)
    save_scored_people(test_df, best_model, best_features)
    save_feature_importance(best_model, best_features, best_key)
    print("Modeling outputs saved.")
    return comparison


def flatten_report(report: dict, experiment: str, model: str) -> pd.DataFrame:
    rows = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            rows.append(
                {
                    "experiment": experiment,
                    "model": model,
                    "class": label,
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1_score": metrics.get("f1-score"),
                    "support": metrics.get("support"),
                }
            )
    return pd.DataFrame(rows)


def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, best_key: tuple[str, str]) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=INCOME_BRACKETS)
    rows = []
    for actual_label, row in zip(INCOME_BRACKETS, matrix):
        for predicted_label, count in zip(INCOME_BRACKETS, row):
            rows.append(
                {
                    "best_experiment": best_key[0],
                    "best_model": best_key[1],
                    "actual_income_bracket": actual_label,
                    "predicted_income_bracket": predicted_label,
                    "count": int(count),
                }
            )
    pd.DataFrame(rows).to_csv(CONFUSION_MATRIX_PATH, index=False)


def save_scored_people(test_df: pd.DataFrame, model: Pipeline, features: list[str]) -> None:
    scored = test_df.copy()
    if len(scored) > MAX_SCORED_EXPORT_ROWS:
        scored = scored.sample(n=MAX_SCORED_EXPORT_ROWS, random_state=RANDOM_STATE).copy()
    predictions = model.predict(scored[features])
    scored["actual_income_bracket"] = scored[TARGET]
    scored["predicted_income_bracket"] = predictions
    scored["prediction_correct_flag"] = (scored["actual_income_bracket"] == scored["predicted_income_bracket"]).astype(int)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scored[features])
        classes = list(model.classes_)
        for class_name in INCOME_BRACKETS:
            if class_name in classes:
                scored[f"probability_{clean_label(class_name)}"] = probabilities[:, classes.index(class_name)]
            else:
                scored[f"probability_{clean_label(class_name)}"] = 0.0
        scored["probability_100000_plus"] = scored["probability_100000_plus"]

    tableau_columns = [
        "actual_income_bracket",
        "predicted_income_bracket",
        "prediction_correct_flag",
        "probability_0",
        "probability_1_24999",
        "probability_25000_49999",
        "probability_50000_74999",
        "probability_75000_99999",
        "probability_100000_plus",
        "statefip",
        "state_name",
        "age",
        "age_group",
        "education_group",
        "degree_field_group",
        "second_degree_field_group",
        "race_group",
        "sex_label",
    ]
    for column in tableau_columns:
        if column not in scored.columns:
            scored[column] = np.nan
    scored[tableau_columns].to_csv(MODEL_SCORED_PATH, index=False)


def save_feature_importance(model: Pipeline, features: list[str], best_key: tuple[str, str]) -> None:
    model_step = model.named_steps["model"]
    preprocess = model.named_steps["preprocess"]
    try:
        feature_names = preprocess.get_feature_names_out(features)
    except Exception:
        feature_names = np.array(features)

    if hasattr(model_step, "feature_importances_"):
        importances = model_step.feature_importances_
        if len(importances) != len(feature_names):
            feature_names = np.array([f"feature_{i}" for i in range(len(importances))])
        output = pd.DataFrame({"feature": feature_names, "importance": importances})
    elif hasattr(model_step, "coef_"):
        coefficients = np.abs(model_step.coef_).mean(axis=0)
        output = pd.DataFrame({"feature": feature_names, "importance": coefficients})
    else:
        output = pd.DataFrame({"feature": features, "importance": np.nan})

    output["experiment"] = best_key[0]
    output["model"] = best_key[1]
    output.sort_values("importance", ascending=False).head(100).to_csv(
        FEATURE_IMPORTANCE_PATH, index=False
    )


def clean_label(label: str) -> str:
    return (
        label.lower()
        .replace("$", "")
        .replace(",", "")
        .replace("-", "_")
        .replace("+", "_plus")
    )


if __name__ == "__main__":
    train_income_models()
