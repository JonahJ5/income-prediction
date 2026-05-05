"""Clean the raw IPUMS USA / ACS extract with pandas."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from config import (
    CHUNK_SIZE,
    CLEAN_DATA_PATH,
    DEFAULT_RAW_DATA_PATH,
    QUALITY_REPORT_PATH,
    ensure_directories,
)
from feature_engineering import add_engineered_features, clean_income_codes, keep_expected_columns, normalize_columns


def resolve_raw_path(raw_path: str | Path | None = None) -> Path:
    """Find the raw ACS extract path."""
    if raw_path:
        return Path(raw_path)
    env_path = os.getenv("ACS_RAW_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_RAW_DATA_PATH


def clean_ipums_data(raw_path: str | Path | None = None) -> pd.DataFrame:
    """Clean the IPUMS extract and write the processed CSV plus quality report."""
    ensure_directories()
    source_path = resolve_raw_path(raw_path)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {source_path}. Set ACS_RAW_PATH or place the file there."
        )

    print(f"Reading raw IPUMS extract from: {source_path}")
    print("Cleaning in chunks so large ACS files can be handled safely...")

    row_counts = {
        "raw_rows": 0,
        "after_age_filter": 0,
        "after_worker_filter": 0,
        "after_income_filter": 0,
    }
    duplicate_key_count = 0
    chunk_reports = []
    seen_person_keys: set[tuple] = set()
    first_write = True

    for chunk_number, chunk in enumerate(pd.read_csv(source_path, chunksize=CHUNK_SIZE, low_memory=False), start=1):
        chunk = normalize_columns(chunk)
        chunk = keep_expected_columns(chunk)
        row_counts["raw_rows"] += len(chunk)

        if "incwage" not in chunk.columns or "age" not in chunk.columns:
            raise ValueError("The extract must include AGE and INCWAGE for this project.")

        chunk["age"] = pd.to_numeric(chunk["age"], errors="coerce")
        chunk["incwage"] = clean_income_codes(chunk["incwage"])
        chunk["perwt"] = pd.to_numeric(chunk.get("perwt", 1), errors="coerce").fillna(1)

        chunk = chunk[chunk["age"].between(18, 65, inclusive="both")].copy()
        row_counts["after_age_filter"] += len(chunk)

        if "empstat" in chunk.columns:
            chunk = chunk[pd.to_numeric(chunk["empstat"], errors="coerce").eq(1)].copy()
        else:
            chunk = chunk[chunk["incwage"].fillna(0).gt(0)].copy()
        row_counts["after_worker_filter"] += len(chunk)

        chunk = chunk[chunk["incwage"].notna() & chunk["incwage"].ge(0)].copy()
        row_counts["after_income_filter"] += len(chunk)

        if {"serial", "pernum"}.issubset(chunk.columns):
            keys = list(zip(chunk["serial"], chunk["pernum"]))
            duplicate_key_count += sum(key in seen_person_keys for key in keys)
            seen_person_keys.update(keys)
            chunk = chunk.drop_duplicates(subset=["serial", "pernum"])
        else:
            duplicate_key_count += int(chunk.duplicated().sum())
            chunk = chunk.drop_duplicates()

        engineered = add_engineered_features(chunk)
        unusual_income = int(engineered["incwage"].gt(500_000).sum())
        chunk_reports.append(
            {
                "chunk": chunk_number,
                "rows_after_cleaning": len(engineered),
                "missing_incwage": int(engineered["incwage"].isna().sum()),
                "missing_age": int(engineered["age"].isna().sum()),
                "zero_income_rows": int(engineered["incwage"].eq(0).sum()),
                "income_over_500k_rows": unusual_income,
            }
        )

        engineered.to_csv(
            CLEAN_DATA_PATH,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )
        first_write = False
        print(f"  Cleaned chunk {chunk_number:,}: {len(engineered):,} rows saved")

    quality_report = build_quality_report(row_counts, duplicate_key_count, chunk_reports)
    quality_report.to_csv(QUALITY_REPORT_PATH, index=False)
    print(f"Clean dataset saved to: {CLEAN_DATA_PATH}")
    print(f"Data quality report saved to: {QUALITY_REPORT_PATH}")
    return quality_report


def build_quality_report(
    row_counts: dict[str, int],
    duplicate_key_count: int,
    chunk_reports: list[dict[str, int]],
) -> pd.DataFrame:
    """Create a compact, Tableau/Excel-friendly quality report."""
    chunk_df = pd.DataFrame(chunk_reports)
    total_clean_rows = int(chunk_df["rows_after_cleaning"].sum()) if not chunk_df.empty else 0
    rows = [
        ("raw_rows", row_counts["raw_rows"], "Rows read from source file"),
        ("after_age_filter", row_counts["after_age_filter"], "Rows age 18 to 65"),
        ("after_worker_filter", row_counts["after_worker_filter"], "Rows after employed/worker filter"),
        ("after_income_filter", row_counts["after_income_filter"], "Rows after income validity checks"),
        ("clean_rows_saved", total_clean_rows, "Rows written to processed clean dataset"),
        ("duplicate_person_keys_removed", duplicate_key_count, "Duplicate SERIAL/PERNUM keys seen"),
        (
            "zero_income_rows",
            int(chunk_df["zero_income_rows"].sum()) if not chunk_df.empty else 0,
            "Clean rows with $0 wage income",
        ),
        (
            "income_over_500k_rows",
            int(chunk_df["income_over_500k_rows"].sum()) if not chunk_df.empty else 0,
            "Potential outlier rows above $500,000",
        ),
        (
            "missing_incwage_after_cleaning",
            int(chunk_df["missing_incwage"].sum()) if not chunk_df.empty else 0,
            "Missing wage income after cleaning",
        ),
        (
            "missing_age_after_cleaning",
            int(chunk_df["missing_age"].sum()) if not chunk_df.empty else 0,
            "Missing age after cleaning",
        ),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "notes"])


if __name__ == "__main__":
    clean_ipums_data()
