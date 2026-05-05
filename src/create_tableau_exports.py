"""Create Tableau-ready summary CSV files from the cleaned ACS dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    CHUNK_SIZE,
    CLEAN_DATA_PATH,
    INCOME_BRACKETS,
    INCOME_DISTRIBUTION_PATH,
    PROFILE_SUMMARY_PATH,
    STATE_SUMMARY_PATH,
    ensure_directories,
)


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna()
    if valid.sum() == 0 or weights[valid].sum() == 0:
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def create_tableau_exports() -> None:
    """Build clean aggregate files for Tableau dashboards."""
    ensure_directories()
    print("Creating Tableau aggregate exports...")
    chunks = pd.read_csv(CLEAN_DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False)

    state_rows = []
    income_distribution_rows = []
    profile_frames = []

    for chunk_number, chunk in enumerate(chunks, start=1):
        numeric_cols = ["perwt", "incwage", "age", "famsize", "bachelors_or_higher_flag"]
        for column in numeric_cols:
            if column in chunk.columns:
                chunk[column] = pd.to_numeric(chunk[column], errors="coerce")

        for state_name, group in chunk.groupby("state_name", dropna=False):
            weight = group["perwt"].fillna(0)
            weighted_population = weight.sum()
            row = {
                "state_name": state_name,
                "statefip": group["statefip"].iloc[0] if "statefip" in group else np.nan,
                "weighted_population_represented": weighted_population,
                "weighted_wage_sum": (group["incwage"].fillna(0) * weight).sum(),
                "weighted_age_sum": (group["age"].fillna(0) * weight).sum(),
                "weighted_family_size_sum": (group["famsize"].fillna(0) * weight).sum(),
                "weighted_bachelors_plus_sum": (
                    group["bachelors_or_higher_flag"].fillna(0) * weight
                ).sum(),
                "unweighted_count": len(group),
                "unweighted_median_wage_income": group["incwage"].median(),
            }
            for bracket in INCOME_BRACKETS:
                row[f"weighted_{clean_label(bracket)}_sum"] = weight[group["income_bracket"].eq(bracket)].sum()
            state_rows.append(row)

            bracket_counts = group.groupby("income_bracket", dropna=False)["perwt"].sum().reset_index()
            bracket_counts["state_name"] = state_name
            income_distribution_rows.append(bracket_counts)

        profile_frames.append(
            summarize_profile(chunk, ["education_group", "income_bracket"], "education_income_profile")
        )
        profile_frames.append(
            summarize_profile(chunk, ["age_group", "income_bracket"], "age_income_profile")
        )
        profile_frames.append(
            summarize_profile(chunk, ["race_group", "income_bracket"], "race_income_profile")
        )
        profile_frames.append(
            summarize_profile(chunk, ["degree_field_group", "income_bracket"], "degree_field_income_profile")
        )
        print(f"  Aggregated chunk {chunk_number:,}")

    state_summary = finalize_state_summary(pd.DataFrame(state_rows))
    state_summary.to_csv(STATE_SUMMARY_PATH, index=False)

    income_distribution = finalize_income_distribution(pd.concat(income_distribution_rows, ignore_index=True))
    income_distribution.to_csv(INCOME_DISTRIBUTION_PATH, index=False)

    profile_summary = pd.concat(profile_frames, ignore_index=True)
    profile_summary = (
        profile_summary.groupby(["profile_type", "dimension_1", "dimension_2"], as_index=False)
        .agg(weighted_population=("weighted_population", "sum"), unweighted_count=("unweighted_count", "sum"))
    )
    profile_summary.to_csv(PROFILE_SUMMARY_PATH, index=False)
    print("Tableau aggregate exports saved.")


def summarize_profile(chunk: pd.DataFrame, columns: list[str], profile_type: str) -> pd.DataFrame:
    summary = (
        chunk.groupby(columns, dropna=False)
        .agg(weighted_population=("perwt", "sum"), unweighted_count=("perwt", "size"))
        .reset_index()
    )
    summary["profile_type"] = profile_type
    summary = summary.rename(columns={columns[0]: "dimension_1", columns[1]: "dimension_2"})
    return summary[["profile_type", "dimension_1", "dimension_2", "weighted_population", "unweighted_count"]]


def finalize_state_summary(raw_state_summary: pd.DataFrame) -> pd.DataFrame:
    grouped = raw_state_summary.groupby(["state_name", "statefip"], as_index=False).agg(
        weighted_population_represented=("weighted_population_represented", "sum"),
        weighted_wage_sum=("weighted_wage_sum", "sum"),
        weighted_age_sum=("weighted_age_sum", "sum"),
        weighted_family_size_sum=("weighted_family_size_sum", "sum"),
        weighted_bachelors_plus_sum=("weighted_bachelors_plus_sum", "sum"),
        unweighted_count=("unweighted_count", "sum"),
        unweighted_median_wage_income=("unweighted_median_wage_income", "median"),
        **{
            f"weighted_{clean_label(bracket)}_sum": (f"weighted_{clean_label(bracket)}_sum", "sum")
            for bracket in INCOME_BRACKETS
        },
    )
    denom = grouped["weighted_population_represented"].replace(0, np.nan)
    grouped["weighted_average_wage_income"] = grouped["weighted_wage_sum"] / denom
    grouped["weighted_average_age"] = grouped["weighted_age_sum"] / denom
    grouped["weighted_average_family_size"] = grouped["weighted_family_size_sum"] / denom
    grouped["bachelors_degree_or_higher_share"] = grouped["weighted_bachelors_plus_sum"] / denom
    for bracket in INCOME_BRACKETS:
        label = clean_label(bracket)
        grouped[f"share_{label}"] = grouped[f"weighted_{label}_sum"] / denom
    drop_cols = [col for col in grouped.columns if col.startswith("weighted_") and col.endswith("_sum")]
    return grouped.drop(columns=drop_cols).sort_values("state_name")


def finalize_income_distribution(raw_distribution: pd.DataFrame) -> pd.DataFrame:
    output = (
        raw_distribution.groupby(["state_name", "income_bracket"], as_index=False)
        .agg(weighted_population=("perwt", "sum"))
    )
    totals = output.groupby("state_name")["weighted_population"].transform("sum").replace(0, np.nan)
    output["state_income_bracket_share"] = output["weighted_population"] / totals
    return output.sort_values(["state_name", "income_bracket"])


def clean_label(label: str) -> str:
    return (
        label.lower()
        .replace("$", "")
        .replace(",", "")
        .replace("-", "_")
        .replace("+", "_plus")
    )


if __name__ == "__main__":
    create_tableau_exports()
