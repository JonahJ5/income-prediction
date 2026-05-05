# Data Dictionary

This project uses only the variables included in the selected IPUMS USA / ACS extract. It does not create placeholders for variables that are not in the extract.

## Raw IPUMS Variables

| Variable | Description | Project use |
|---|---|---|
| `YEAR` | Census year | Dataset context |
| `MULTYEAR` | Actual year of survey, multi-year ACS/PRCS | Dataset context |
| `SAMPLE` | IPUMS sample identifier | Dataset metadata |
| `SERIAL` | Household serial number | Duplicate person key with `PERNUM` |
| `CBSERIAL` | Original Census Bureau household serial number | Metadata |
| `HHWT` | Household weight | Household context |
| `CLUSTER` | Household cluster for variance estimation | Metadata |
| `STRATA` | Household strata for variance estimation | Metadata |
| `GQ` | Group quarters status | Context and quality checks |
| `STATEFIP` | State FIPS code | State summaries and dashboard geography |
| `COUNTYFIP` | County FIPS code | County context when identifiable |
| `PERNUM` | Person number in sample unit | Duplicate person key with `SERIAL` |
| `PERWT` | Person weight | Weighted state-level summaries |
| `FAMSIZE` | Number of own family members in household | Model feature and weighted summaries |
| `SEX` | Sex | Model feature and readable label |
| `AGE` | Age | Filter, model feature, and age groups |
| `MARST` | Marital status | Model feature and readable label |
| `RACE` | Race | Model feature and readable label |
| `CITIZEN` | Citizenship status | Model feature and readable label |
| `EDUC` | Educational attainment | Education grouping and bachelor's-plus flag |
| `DEGFIELD` | Field of degree | Broad degree field grouping |
| `DEGFIELD2` | Field of degree 2 | Broad second degree field grouping |
| `EMPSTAT` | Employment status | Worker-focused filter and model feature |
| `INCWAGE` | Wage and salary income | Main target source |

## Cleaned Variables

| Variable | Description |
|---|---|
| `incwage` | Numeric wage income after invalid and special missing codes are handled |
| `age` | Numeric age |
| `perwt` | Numeric person weight |
| `famsize` | Numeric family size |
| `state_name` | State name mapped from `statefip` |
| `sex_label` | Readable sex label from `SEX` |
| `marital_status` | Readable marital status from `MARST` |
| `citizenship` | Readable citizenship category from `CITIZEN` |
| `employment_status` | Readable employment status from `EMPSTAT` |
| `race_group` | Race label from `RACE` only; Hispanic origin is not present in this extract |

## Engineered Variables

| Variable | Description |
|---|---|
| `age_group` | Age bucket: `18-24`, `25-34`, `35-44`, `45-54`, `55-65` |
| `log_incwage` | `log1p(incwage)` for skew-aware analysis |
| `education_group` | Broad education grouping from `EDUC` |
| `degree_field_group` | Broad field-of-degree grouping from `DEGFIELD` |
| `second_degree_field_group` | Broad field-of-degree grouping from `DEGFIELD2` |
| `bachelors_or_higher_flag` | `1` for bachelor's degree or graduate degree based on `EDUC` |

## Target Variable

| Variable | Description |
|---|---|
| `income_bracket` | Prediction target derived from `INCWAGE` |

Income bracket values:

| Bracket | Rule |
|---|---|
| `$0` | `INCWAGE == 0` |
| `$1-$24,999` | `1 <= INCWAGE <= 24,999` |
| `$25,000-$49,999` | `25,000 <= INCWAGE <= 49,999` |
| `$50,000-$74,999` | `50,000 <= INCWAGE <= 74,999` |
| `$75,000-$99,999` | `75,000 <= INCWAGE <= 99,999` |
| `$100,000+` | `INCWAGE >= 100,000` |

## Tableau Export Datasets

| File | Grain | Description |
|---|---|---|
| `state_income_summary.csv` | One row per state | Weighted population, average wage income, median wage income, weighted average age, weighted average family size, bachelor's-plus share, and income bracket shares |
| `income_distribution_by_state.csv` | State by income bracket | Weighted population and share for each bracket in each state |
| `model_scored_people.csv` | Scored test person row | Actual bracket, predicted bracket, correctness flag, probabilities, and available dashboard dimensions |
| `feature_importance.csv` | Model feature | Feature importance or coefficient strength for the selected best model |
| `model_comparison.csv` | Model experiment | Accuracy, macro F1, weighted F1, and row counts for full and reduced models |
| `profile_summary.csv` | Profile category by bracket | Weighted profile summaries for education, age, race, and degree field |
