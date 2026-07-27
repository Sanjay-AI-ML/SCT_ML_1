"""
data_preprocessing.py
----------------------
Loads the Ames Housing / Kaggle "House Prices: Advanced Regression Techniques"
dataset and engineers the feature set used by the models in this project.

Author: Sanjay
Project: House Price Prediction — Linear Regression (SkillCraft ML Internship, Task 01)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "train.csv"


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw Kaggle House Prices training data."""
    df = pd.read_csv(path)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature set for the project.

    Core required features (per the task brief):
        - Square footage  -> GrLivArea (above-grade living area, sq ft)
        - Bedrooms         -> BedroomAbvGr
        - Bathrooms        -> engineered as full baths + 0.5 * half baths
                               (combines above-grade AND basement baths,
                                which is how real-estate listings define
                                "total bathrooms")

    A few extra engineered features are included for the "enhanced" model
    (kept separate from the core 3-feature model so the required deliverable
    stays exactly as specified):
        - TotalSF          -> total finished square footage (basement + floors)
        - HouseAge         -> age of the house at time of sale
        - OverallQual      -> Kaggle's built-in 1-10 quality rating
    """
    data = df.copy()

    # ---- Target ----
    # SalePrice is right-skewed (typical of price data). We keep the raw
    # target for interpretability (dollars) but also expose a log-target
    # option, which is standard practice and improves linear model fit.
    data["SalePrice"] = data["SalePrice"].astype(float)

    # ---- Core required features ----
    data["SquareFootage"] = data["GrLivArea"].astype(float)
    data["Bedrooms"] = data["BedroomAbvGr"].astype(float)
    data["Bathrooms"] = (
        data["FullBath"].astype(float)
        + 0.5 * data["HalfBath"].astype(float)
        + data["BsmtFullBath"].astype(float)
        + 0.5 * data["BsmtHalfBath"].astype(float)
    )

    # ---- Extra engineered features (used only in the enhanced model) ----
    data["TotalSF"] = (
        data["TotalBsmtSF"].astype(float)
        + data["1stFlrSF"].astype(float)
        + data["2ndFlrSF"].astype(float)
    )
    data["HouseAge"] = data["YrSold"].astype(float) - data["YearBuilt"].astype(float)
    data["HouseAge"] = data["HouseAge"].clip(lower=0)  # guard against data entry quirks
    data["Remodeled"] = (data["YearRemodAdd"] != data["YearBuilt"]).astype(int)
    data["GarageCars"] = data["GarageCars"].astype(float)
    data["OverallQual"] = data["OverallQual"].astype(float)

    return data


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the two well-documented outliers in this dataset:
    two houses with GrLivArea > 4000 sq ft that sold for unusually LOW
    prices (likely distressed / partial sales). This is a widely-known
    data-cleaning step for this exact dataset (see Kaggle community
    notebooks) and meaningfully improves a linear model's fit.
    """
    cleaned = df[~((df["SquareFootage"] > 4000) & (df["SalePrice"] < 300000))].copy()
    removed = len(df) - len(cleaned)
    return cleaned, removed


def get_core_dataset(path: Path = DATA_PATH):
    """
    Returns the cleaned dataframe restricted to the columns needed for the
    CORE required model: SquareFootage, Bedrooms, Bathrooms -> SalePrice.
    """
    raw = load_raw_data(path)
    engineered = engineer_features(raw)
    cleaned, n_removed = remove_outliers(engineered)

    core_cols = ["SquareFootage", "Bedrooms", "Bathrooms", "SalePrice"]
    return cleaned[core_cols].reset_index(drop=True), n_removed


def get_enhanced_dataset(path: Path = DATA_PATH):
    """
    Returns the cleaned dataframe with the extended feature set used for
    the enhanced / comparison model.
    """
    raw = load_raw_data(path)
    engineered = engineer_features(raw)
    cleaned, n_removed = remove_outliers(engineered)

    enhanced_cols = [
        "SquareFootage", "Bedrooms", "Bathrooms",
        "TotalSF", "OverallQual", "GarageCars", "HouseAge", "Remodeled",
        "SalePrice",
    ]
    return cleaned[enhanced_cols].reset_index(drop=True), n_removed


if __name__ == "__main__":
    core_df, removed = get_core_dataset()
    print(f"Removed {removed} outlier rows.")
    print(f"Core dataset shape: {core_df.shape}")
    print(core_df.head())
    print(core_df.describe())
