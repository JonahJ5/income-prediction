"""Feature engineering helpers for the selected IPUMS USA / ACS extract."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import INCOME_BRACKETS


STATE_FIPS = {
    1: "Alabama",
    2: "Alaska",
    4: "Arizona",
    5: "Arkansas",
    6: "California",
    8: "Colorado",
    9: "Connecticut",
    10: "Delaware",
    11: "District of Columbia",
    12: "Florida",
    13: "Georgia",
    15: "Hawaii",
    16: "Idaho",
    17: "Illinois",
    18: "Indiana",
    19: "Iowa",
    20: "Kansas",
    21: "Kentucky",
    22: "Louisiana",
    23: "Maine",
    24: "Maryland",
    25: "Massachusetts",
    26: "Michigan",
    27: "Minnesota",
    28: "Mississippi",
    29: "Missouri",
    30: "Montana",
    31: "Nebraska",
    32: "Nevada",
    33: "New Hampshire",
    34: "New Jersey",
    35: "New Mexico",
    36: "New York",
    37: "North Carolina",
    38: "North Dakota",
    39: "Ohio",
    40: "Oklahoma",
    41: "Oregon",
    42: "Pennsylvania",
    44: "Rhode Island",
    45: "South Carolina",
    46: "South Dakota",
    47: "Tennessee",
    48: "Texas",
    49: "Utah",
    50: "Vermont",
    51: "Virginia",
    53: "Washington",
    54: "West Virginia",
    55: "Wisconsin",
    56: "Wyoming",
    72: "Puerto Rico",
}


EXPECTED_RAW_COLUMNS = {
    "year",
    "multyear",
    "sample",
    "serial",
    "cbserial",
    "hhwt",
    "cluster",
    "strata",
    "gq",
    "statefip",
    "countyfip",
    "pernum",
    "perwt",
    "famsize",
    "sex",
    "age",
    "marst",
    "race",
    "citizen",
    "educ",
    "degfield",
    "degfield2",
    "empstat",
    "incwage",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lower-case, snake-style column names."""
    cleaned = df.copy()
    cleaned.columns = (
        cleaned.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return cleaned


def keep_expected_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only variables that are part of this IPUMS extract."""
    available = [column for column in df.columns if column in EXPECTED_RAW_COLUMNS]
    return df[available].copy()


def clean_income_codes(series: pd.Series) -> pd.Series:
    """Convert INCWAGE to numeric dollars and remove common IPUMS missing codes."""
    income = pd.to_numeric(series, errors="coerce")
    income = income.where(income >= 0)
    income = income.mask(income >= 9_999_998)
    return income


def age_group(age: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(age, errors="coerce"),
        bins=[17, 24, 34, 44, 54, 65],
        labels=["18-24", "25-34", "35-44", "45-54", "55-65"],
    ).astype("object").fillna("Unknown")


def income_bracket(income: pd.Series) -> pd.Series:
    bins = [-0.01, 0, 24_999, 49_999, 74_999, 99_999, np.inf]
    return pd.cut(income, bins=bins, labels=INCOME_BRACKETS).astype("object")


def sex_label(value: object) -> str:
    return {1: "Male", 2: "Female"}.get(_safe_int(value), "Unknown")


def marital_status_label(value: object) -> str:
    mapping = {
        1: "Married, spouse present",
        2: "Married, spouse absent",
        3: "Separated",
        4: "Divorced",
        5: "Widowed",
        6: "Never married",
    }
    return mapping.get(_safe_int(value), "Unknown")


def citizenship_label(value: object) -> str:
    mapping = {
        0: "N/A",
        1: "Born abroad of American parents",
        2: "Naturalized citizen",
        3: "Not a citizen",
        4: "Born in U.S.",
        5: "Born in U.S. outlying area",
    }
    return mapping.get(_safe_int(value), "Unknown")


def employment_status_label(value: object) -> str:
    return {0: "N/A", 1: "Employed", 2: "Unemployed", 3: "Not in labor force"}.get(
        _safe_int(value), "Unknown"
    )


def education_group(value: object) -> str:
    """Broad education group from EDUC, the education field available here."""
    educ = _safe_int(value)
    if educ is None:
        return "Unknown"
    if educ <= 5:
        return "Less than high school"
    if educ == 6:
        return "High school or GED"
    if educ in {7, 8, 9}:
        return "Some college or associate"
    if educ == 10:
        return "Bachelor's degree"
    if educ >= 11:
        return "Graduate degree"
    return "Unknown"


def race_group(value: object) -> str:
    """Race category from RACE only. Hispanic origin is not in this extract."""
    mapping = {
        1: "White",
        2: "Black/African American",
        3: "American Indian or Alaska Native",
        4: "Chinese",
        5: "Japanese",
        6: "Other Asian or Pacific Islander",
        7: "Other race",
        8: "Two major races",
        9: "Three or more races",
    }
    return mapping.get(_safe_int(value), "Unknown")


def degree_field_group(value: object) -> str:
    """Broad field-of-degree grouping from DEGFIELD/DEGFIELD2."""
    code = _safe_int(value)
    if code is None or code == 0:
        return "No degree field reported"
    if 11 <= code <= 24:
        return "STEM"
    if 25 <= code <= 36:
        return "Health, education, social sciences"
    if 37 <= code <= 49:
        return "Business, communications, public affairs"
    if 50 <= code <= 64:
        return "Humanities, arts, other"
    return "Other degree field"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add labels, target buckets, and model/dashboard features."""
    output = keep_expected_columns(normalize_columns(df))
    output["incwage"] = clean_income_codes(output["incwage"])
    output["age"] = pd.to_numeric(output["age"], errors="coerce")
    output["perwt"] = pd.to_numeric(output.get("perwt", 1), errors="coerce").fillna(1)
    output["famsize"] = pd.to_numeric(output.get("famsize"), errors="coerce")

    output["state_name"] = output["statefip"].map(lambda x: STATE_FIPS.get(_safe_int(x), "Unknown"))
    output["age_group"] = age_group(output["age"])
    output["income_bracket"] = income_bracket(output["incwage"]).fillna("Unknown")
    output["log_incwage"] = np.log1p(output["incwage"].clip(lower=0))

    output["sex_label"] = output.get("sex", pd.Series(index=output.index)).map(sex_label)
    output["marital_status"] = output.get("marst", pd.Series(index=output.index)).map(marital_status_label)
    output["citizenship"] = output.get("citizen", pd.Series(index=output.index)).map(citizenship_label)
    output["employment_status"] = output.get("empstat", pd.Series(index=output.index)).map(employment_status_label)
    output["education_group"] = output.get("educ", pd.Series(index=output.index)).map(education_group)
    output["race_group"] = output.get("race", pd.Series(index=output.index)).map(race_group)
    output["degree_field_group"] = output.get("degfield", pd.Series(index=output.index)).map(degree_field_group)
    output["second_degree_field_group"] = output.get("degfield2", pd.Series(index=output.index)).map(
        degree_field_group
    )
    output["bachelors_or_higher_flag"] = output["education_group"].isin(
        ["Bachelor's degree", "Graduate degree"]
    ).astype(int)
    return output


def _safe_int(value: object) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None
