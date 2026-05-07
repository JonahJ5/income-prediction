# ACS Income Prediction & State Demographic Explorer

This project uses an IPUMS USA / ACS microdata extract to clean person-level records, engineer analysis features, train Python-based wage income bracket prediction models, and export Tableau-ready CSV files for a state demographic dashboard.

This project explores associations between demographic, education, employment, and geographic characteristics and wage income. It is not a causal model and should not be used for individual decision-making such as hiring, lending, or eligibility decisions.

## Business and Analytics Question

How do age, education, employment status, family size, degree field, race, sex, citizenship, marital status, and geography relate to wage income brackets across U.S. states, and how well can common machine-learning classifiers predict a person's wage income bracket from the fields available in this ACS extract?

## Dataset

The source data is an IPUMS USA / ACS CSV microdata extract. This project intentionally uses only the fields included in the extract:

`YEAR`, `MULTYEAR`, `SAMPLE`, `SERIAL`, `CBSERIAL`, `HHWT`, `CLUSTER`, `STRATA`, `GQ`, `STATEFIP`, `COUNTYFIP`, `PERNUM`, `PERWT`, `FAMSIZE`, `SEX`, `AGE`, `MARST`, `RACE`, `CITIZEN`, `EDUC`, `DEGFIELD`, `DEGFIELD2`, `EMPSTAT`, and `INCWAGE`.

Raw ACS/IPUMS extracts can be very large, so raw files are intentionally excluded from Git. The pipeline currently points to:

```text
C:\Users\jutzi\Downloads\usa_00002.csv\usa_00002.csv
```

You can override that path with an environment variable:

```powershell
$env:ACS_RAW_PATH = "C:\path\to\your\ipums_extract.csv"
python src/run_pipeline.py
```

## Why Python and Tableau

Python is used for all cleaning, calculations, feature engineering, model training, scoring, and exports. Tableau is used only as the presentation and dashboard layer. The dashboard does not depend on Tableau's built-in predictive tools.

## Project Structure

```text
acs-income-prediction-dashboard/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- data_dictionary.md
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- tableau_exports/
|-- notebooks/
|-- src/
|-- models/
`-- outputs/
```

## Cleaning Steps

The cleaning script:

- Reads the large raw CSV in pandas chunks.
- Standardizes column names to lower-case snake case.
- Keeps only the variables in this extract.
- Filters to working-age adults from 18 to 65.
- Creates a worker-focused dataset using `EMPSTAT == 1`.
- Converts `INCWAGE`, `AGE`, `PERWT`, and `FAMSIZE` to numeric fields.
- Handles invalid, negative, missing, and special IPUMS income codes.
- Removes duplicate person keys using `SERIAL` and `PERNUM`.
- Creates readable categories for state, sex, education, citizenship, employment status, marital status, race, and degree field.
- Saves a data quality report with row counts, duplicate checks, missing values, income checks, and unusual values.

Main cleaned output:

```text
data/processed/acs_income_clean.csv
outputs/metrics/data_quality_report.csv
```

## Feature Engineering

Python creates these fields from the available extract variables:

- `age_group`
- `income_bracket`
- `log_incwage`
- `education_group`
- `degree_field_group`
- `second_degree_field_group`
- `race_group`
- `state_name`
- `bachelors_or_higher_flag`

Income brackets:

- `$0`
- `$1-$24,999`
- `$25,000-$49,999`
- `$50,000-$74,999`
- `$75,000-$99,999`
- `$100,000+`

## Weighted State-Level Summaries

The Tableau state summary uses `PERWT` to calculate:

- Weighted population represented
- Weighted average wage income
- Unweighted median wage income
- Weighted average age
- Weighted average family size
- Share in each income bracket
- Share with bachelor's degree or higher

## Modeling Approach

The target variable is `income_bracket`, derived from `INCWAGE`.

The pipeline trains:

- Logistic Regression baseline
- Random Forest classifier
- HistGradientBoostingClassifier

Features include:

- `state_name`
- `age`
- `famsize`
- `age_group`
- `sex_label`
- `race_group`
- `marital_status`
- `citizenship`
- `education_group`
- `employment_status`
- `degree_field_group`
- `second_degree_field_group`

Model 1 uses all available features. Model 2 excludes `race_group` and `sex_label` to compare performance without those sensitive demographic fields. Because Hispanic origin is not present in this extract, the project does not build a combined race/ethnicity field.

Evaluation includes accuracy, macro F1, weighted F1, classification reports, a confusion matrix, model comparison, and feature importance where practical.

## Tableau Dashboard Plan

Tableau dashboard pages:

- State Income Overview: map and ranking of weighted average wage income, median wage income, and population represented.
- Income Bracket Mix: stacked bar or heatmap showing bracket shares by state.
- Demographic Explorer: filters for age group, education, race, sex, citizenship, and degree field.
- Model Performance: confusion matrix, accuracy/F1 table, prediction correctness rate, and probability distribution.
- Model Signals: feature importance table or bar chart.

Tableau export files:

```text
data/tableau_exports/state_income_summary.csv
data/tableau_exports/model_scored_people.csv
data/tableau_exports/feature_importance.csv
data/tableau_exports/model_comparison.csv
data/tableau_exports/income_distribution_by_state.csv
data/tableau_exports/profile_summary.csv
```

## Ethics and Limitations

This is an exploratory and educational project. ACS income patterns are shaped by complex historical, geographic, labor-market, policy, and measurement factors. Model predictions should be interpreted as pattern recognition within survey data, not as explanations of individual worth or causal effects.

Sensitive demographic variables can improve measured prediction performance while raising fairness and misuse concerns. For that reason, the project compares a model using all available features with a second model that excludes race and sex. This comparison does not guarantee fairness; it simply shows how performance changes when those fields are removed.

Other limitations:

- ACS values are survey estimates and use person weights.
- `INCWAGE` is self-reported wage and salary income, not total income.
- The extract does not include hours worked, weeks worked, occupation, industry, class of worker, Hispanic origin, or detailed code variables.
- Broad category groupings simplify complex IPUMS codes.
- The model is trained on a sample by default to keep local runtime manageable.

## How to Run

Create and activate a Python environment, then install requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the full pipeline:

```powershell
python src/run_pipeline.py
```

The pipeline prints progress messages and writes processed datasets, model metrics, model artifacts, and Tableau-ready CSV files.
