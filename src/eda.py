"""
eda.py
-------
Exploratory Data Analysis for the House Price Prediction project.
Generates all the visual + statistical groundwork that justifies the
modeling choices made in train_model.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from data_preprocessing import get_core_dataset, load_raw_data, engineer_features

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 120

BASE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = BASE_DIR / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_target_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.histplot(df["SalePrice"], kde=True, ax=axes[0], color="#2E86AB", bins=40)
    axes[0].set_title(f"SalePrice Distribution (Skew = {df['SalePrice'].skew():.2f})", fontweight="bold")
    axes[0].set_xlabel("Sale Price ($)")
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

    log_price = np.log1p(df["SalePrice"])
    sns.histplot(log_price, kde=True, ax=axes[1], color="#F18F01", bins=40)
    axes[1].set_title(f"log(SalePrice) Distribution (Skew = {log_price.skew():.2f})", fontweight="bold")
    axes[1].set_xlabel("log(Sale Price + 1)")

    fig.suptitle("Target Variable Analysis: Sale Price is Right-Skewed", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_target_distribution.png", bbox_inches="tight")
    plt.close()


def plot_feature_relationships(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sns.regplot(data=df, x="SquareFootage", y="SalePrice", ax=axes[0],
                scatter_kws={"alpha": 0.4, "s": 25, "color": "#2E86AB"},
                line_kws={"color": "red"})
    axes[0].set_title("Sale Price vs. Square Footage", fontweight="bold")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

    sns.boxplot(data=df, x="Bedrooms", y="SalePrice", hue="Bedrooms", ax=axes[1], palette="crest", legend=False)
    axes[1].set_title("Sale Price by Bedroom Count", fontweight="bold")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

    sns.boxplot(data=df, x="Bathrooms", y="SalePrice", hue="Bathrooms", ax=axes[2], palette="magma", legend=False)
    axes[2].set_title("Sale Price by Bathroom Count", fontweight="bold")
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    axes[2].tick_params(axis="x", rotation=45)

    fig.suptitle("How the Three Core Features Relate to Sale Price", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_feature_relationships.png", bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix — Core Features", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_correlation_heatmap.png", bbox_inches="tight")
    plt.close()


def plot_multicollinearity_explainer(df):
    """
    Specifically illustrates WHY the Bedrooms coefficient turns negative
    in the multiple regression — a key insight for the report / viva.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.scatterplot(data=df, x="Bedrooms", y="SquareFootage", ax=axes[0],
                     alpha=0.4, color="#A23B72", s=30)
    axes[0].set_title("Bedrooms vs. Square Footage\n(correlated predictors → multicollinearity)",
                       fontweight="bold")

    sns.barplot(data=df, x="Bedrooms", y="SalePrice", hue="Bedrooms", ax=axes[1], palette="viridis", legend=False,
                errorbar=("ci", 95))
    axes[1].set_title("Average Sale Price by Bedroom Count\n(bedrooms alone ≠ higher price)",
                       fontweight="bold")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

    fig.suptitle("Why the Bedrooms Coefficient Is Negative in the Multiple Regression",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_multicollinearity_explainer.png", bbox_inches="tight")
    plt.close()


def plot_neighborhood_context(raw_df):
    """Extra context plot using a feature outside the core 3 — shows broader
    data understanding without changing the core required model."""
    top_neighborhoods = raw_df.groupby("Neighborhood")["SalePrice"].median().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=top_neighborhoods.values, y=top_neighborhoods.index, hue=top_neighborhoods.index, palette="rocket", ax=ax, legend=False)
    ax.set_title("Top 10 Neighborhoods by Median Sale Price\n(context: location matters beyond sqft/beds/baths)",
                 fontweight="bold")
    ax.set_xlabel("Median Sale Price ($)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_neighborhood_context.png", bbox_inches="tight")
    plt.close()


def print_summary_stats(df):
    print("=" * 70)
    print("SUMMARY STATISTICS — Core Features")
    print("=" * 70)
    print(df.describe().round(2))
    print("\nCorrelation with SalePrice:")
    print(df.corr()["SalePrice"].sort_values(ascending=False).round(3))
    print(f"\nSalePrice skewness: {df['SalePrice'].skew():.3f}")
    print(f"SalePrice kurtosis: {df['SalePrice'].kurt():.3f}")

    # Shapiro-Wilk normality test (sampled, since test caps at 5000 obs)
    sample = df["SalePrice"].sample(min(len(df), 500), random_state=42)
    stat, p = stats.shapiro(sample)
    print(f"\nShapiro-Wilk normality test on SalePrice: W={stat:.4f}, p={p:.2e}")
    print("  -> " + ("Price is NOT normally distributed (p < 0.05)" if p < 0.05
                      else "Price appears normally distributed (p >= 0.05)"))


if __name__ == "__main__":
    core_df, _ = get_core_dataset()
    raw_df = load_raw_data()

    print_summary_stats(core_df)

    plot_target_distribution(core_df)
    plot_feature_relationships(core_df)
    plot_correlation_heatmap(core_df)
    plot_multicollinearity_explainer(core_df)
    plot_neighborhood_context(raw_df)

    print(f"\nAll EDA figures saved to {FIG_DIR}")
