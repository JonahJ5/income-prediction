"""Project configuration for the ACS income prediction pipeline."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The IPUMS extract lives outside the repo so raw data is not accidentally
# committed. Override this path with ACS_RAW_PATH when needed.
DEFAULT_RAW_DATA_PATH = Path(
    r"C:\Users\jutzi\Downloads\usa_00002.csv\usa_00002.csv"
)

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TABLEAU_EXPORT_DIR = DATA_DIR / "tableau_exports"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
VISUALS_DIR = OUTPUT_DIR / "visuals"
MODELS_DIR = PROJECT_ROOT / "models"

CLEAN_DATA_PATH = PROCESSED_DIR / "acs_income_clean.csv"
QUALITY_REPORT_PATH = METRICS_DIR / "data_quality_report.csv"

STATE_SUMMARY_PATH = TABLEAU_EXPORT_DIR / "state_income_summary.csv"
MODEL_SCORED_PATH = TABLEAU_EXPORT_DIR / "model_scored_people.csv"
FEATURE_IMPORTANCE_PATH = TABLEAU_EXPORT_DIR / "feature_importance.csv"
MODEL_COMPARISON_EXPORT_PATH = TABLEAU_EXPORT_DIR / "model_comparison.csv"
INCOME_DISTRIBUTION_PATH = TABLEAU_EXPORT_DIR / "income_distribution_by_state.csv"
PROFILE_SUMMARY_PATH = TABLEAU_EXPORT_DIR / "profile_summary.csv"

MODEL_COMPARISON_PATH = METRICS_DIR / "model_comparison.csv"
CLASSIFICATION_REPORT_PATH = METRICS_DIR / "classification_report.csv"
CONFUSION_MATRIX_PATH = METRICS_DIR / "confusion_matrix.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_income_bracket_model.joblib"

RANDOM_STATE = 42
CHUNK_SIZE = 250_000
MAX_MODEL_ROWS = 200_000
MAX_SCORED_EXPORT_ROWS = 50_000

INCOME_BRACKETS = [
    "$0",
    "$1-$24,999",
    "$25,000-$49,999",
    "$50,000-$74,999",
    "$75,000-$99,999",
    "$100,000+",
]


def ensure_directories() -> None:
    """Create all project output folders if they do not already exist."""
    for path in [
        RAW_DIR,
        PROCESSED_DIR,
        TABLEAU_EXPORT_DIR,
        METRICS_DIR,
        VISUALS_DIR,
        MODELS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
