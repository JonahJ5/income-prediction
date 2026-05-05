"""Run the complete ACS income prediction project pipeline."""

from clean_ipums import clean_ipums_data
from create_tableau_exports import create_tableau_exports
from train_model import train_income_models


def main() -> None:
    print("Starting ACS Income Prediction & State Demographic Explorer pipeline")
    print("Step 1 of 4: cleaning raw IPUMS data")
    clean_ipums_data()

    print("Step 2 of 4: engineered features were created during cleaning")
    print("Step 3 of 4: training and scoring income-bracket models")
    train_income_models()

    print("Step 4 of 4: creating Tableau-ready aggregate exports")
    create_tableau_exports()
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
