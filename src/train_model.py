"""
train_model.py
----------------
Trains and evaluates:
  1. The CORE required model: Linear Regression on
     SquareFootage + Bedrooms + Bathrooms -> SalePrice
  2. A set of benchmark models (Ridge, Lasso, Random Forest, Gradient
     Boosting) on an enhanced feature set, to contextualize how far a
     simple linear model gets vs. more powerful approaches.

Produces:
  - outputs/models/*.joblib          (saved fitted models)
  - outputs/metrics.json             (all evaluation metrics)
  - outputs/figures/*.png            (diagnostic plots)
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from data_preprocessing import get_core_dataset, get_enhanced_dataset

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 120

BASE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = BASE_DIR / "outputs" / "figures"
MODEL_DIR = BASE_DIR / "outputs" / "models"
METRICS_PATH = BASE_DIR / "outputs" / "metrics.json"
RANDOM_STATE = 42

FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(y_true, y_pred) -> dict:
    """Standard regression metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE_%": mape}


# ------------------------------------------------------------------
# 1. CORE REQUIRED MODEL
# ------------------------------------------------------------------
def train_core_model():
    print("=" * 70)
    print("CORE MODEL — Linear Regression: SquareFootage + Bedrooms + Bathrooms")
    print("=" * 70)

    df, n_removed = get_core_dataset()
    print(f"Rows after cleaning (removed {n_removed} outliers): {len(df)}")

    X = df[["SquareFootage", "Bedrooms", "Bathrooms"]]
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_metrics = evaluate(y_train, y_pred_train)
    test_metrics = evaluate(y_test, y_pred_test)

    # 5-fold cross-validation for a more robust performance estimate
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_r2 = cross_val_score(model, X, y, cv=kf, scoring="r2")
    cv_rmse = -cross_val_score(
        model, X, y, cv=kf, scoring="neg_root_mean_squared_error"
    )

    print("\nModel coefficients:")
    for name, coef in zip(X.columns, model.coef_):
        print(f"  {name:15s}: {coef:>12,.2f}  (price change per +1 unit)")
    print(f"  {'Intercept':15s}: {model.intercept_:>12,.2f}")

    print("\nTrain metrics:", {k: round(v, 4) for k, v in train_metrics.items()})
    print("Test metrics :", {k: round(v, 4) for k, v in test_metrics.items()})
    print(f"\n5-Fold CV  R²  : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    print(f"5-Fold CV  RMSE: ${cv_rmse.mean():,.2f} ± ${cv_rmse.std():,.2f}")

    joblib.dump(model, MODEL_DIR / "core_linear_regression.joblib")

    # -------------------- Diagnostic plots --------------------
    _plot_actual_vs_predicted(
        y_test, y_pred_test,
        "Core Model: Actual vs. Predicted Sale Price",
        FIG_DIR / "core_actual_vs_predicted.png",
    )
    _plot_residuals(
        y_pred_test, y_test - y_pred_test,
        "Core Model: Residual Plot",
        FIG_DIR / "core_residuals.png",
    )
    _plot_coefficients(X.columns, model.coef_, FIG_DIR / "core_coefficients.png")

    return {
        "model": model,
        "coefficients": dict(zip(X.columns, model.coef_.tolist())),
        "intercept": float(model.intercept_),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "cv_r2_mean": float(cv_r2.mean()),
        "cv_r2_std": float(cv_r2.std()),
        "cv_rmse_mean": float(cv_rmse.mean()),
        "cv_rmse_std": float(cv_rmse.std()),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "outliers_removed": n_removed,
    }


# ------------------------------------------------------------------
# 2. ENHANCED / BENCHMARK MODELS  (shows initiative beyond the brief)
# ------------------------------------------------------------------
def train_enhanced_models():
    print("\n" + "=" * 70)
    print("ENHANCED MODELS — extended features, multiple algorithms")
    print("=" * 70)

    df, _ = get_enhanced_dataset()
    feature_cols = [c for c in df.columns if c != "SalePrice"]
    X = df[feature_cols]
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression (extended features)": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "Lasso Regression": Lasso(alpha=100.0, random_state=RANDOM_STATE, max_iter=10000),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE
        ),
    }

    results = {}
    predictions = {}
    for name, m in models.items():
        if "Forest" in name or "Boosting" in name:
            m.fit(X_train, y_train)
            preds = m.predict(X_test)
        else:
            m.fit(X_train_scaled, y_train)
            preds = m.predict(X_test_scaled)
        metrics = evaluate(y_test, preds)
        results[name] = metrics
        predictions[name] = preds
        joblib.dump(m, MODEL_DIR / f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.joblib")
        print(f"{name:40s} R2={metrics['R2']:.4f}  RMSE=${metrics['RMSE']:,.0f}  MAE=${metrics['MAE']:,.0f}")

    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

    _plot_model_comparison(results, FIG_DIR / "model_comparison.png")

    # Feature importance from Random Forest (most reliable importances)
    rf = models["Random Forest"]
    _plot_feature_importance(feature_cols, rf.feature_importances_, FIG_DIR / "feature_importance.png")

    return results


# ------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------
def _plot_actual_vs_predicted(y_true, y_pred, title, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolor="white", s=45, color="#2E86AB")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=2, label="Perfect Prediction")
    ax.set_xlabel("Actual Sale Price ($)")
    ax.set_ylabel("Predicted Sale Price ($)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _plot_residuals(y_pred, residuals, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.5, edgecolor="white", s=40, color="#A23B72")
    axes[0].axhline(0, color="red", linestyle="--", lw=2)
    axes[0].set_xlabel("Predicted Sale Price ($)")
    axes[0].set_ylabel("Residual (Actual − Predicted)")
    axes[0].set_title("Residuals vs. Predicted", fontweight="bold")
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

    sns.histplot(residuals, kde=True, ax=axes[1], color="#F18F01")
    axes[1].set_xlabel("Residual ($)")
    axes[1].set_title("Distribution of Residuals", fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _plot_coefficients(names, coefs, save_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#2E86AB" if c >= 0 else "#C73E1D" for c in coefs]
    ax.barh(list(names), coefs, color=colors)
    ax.set_xlabel("Coefficient (impact on Sale Price, $)")
    ax.set_title("Core Model: Feature Coefficients", fontweight="bold")
    ax.axvline(0, color="black", lw=0.8)
    for i, v in enumerate(coefs):
        ax.text(v, i, f" ${v:,.0f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _plot_model_comparison(results: dict, save_path):
    names = list(results.keys())
    r2_scores = [results[n]["R2"] for n in names]
    rmse_scores = [results[n]["RMSE"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    bars1 = axes[0].barh(names, r2_scores, color=sns.color_palette("viridis", len(names)))
    axes[0].set_xlabel("R² Score (higher is better)")
    axes[0].set_title("Model Comparison — R²", fontweight="bold")
    axes[0].set_xlim(0, 1)
    for bar, val in zip(bars1, r2_scores):
        axes[0].text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=9)

    bars2 = axes[1].barh(names, rmse_scores, color=sns.color_palette("magma", len(names)))
    axes[1].set_xlabel("RMSE ($, lower is better)")
    axes[1].set_title("Model Comparison — RMSE", fontweight="bold")
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    for bar, val in zip(bars2, rmse_scores):
        axes[1].text(val, bar.get_y() + bar.get_height()/2, f" ${val:,.0f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _plot_feature_importance(names, importances, save_path):
    order = np.argsort(importances)[::-1]
    names_sorted = [names[i] for i in order]
    imp_sorted = [importances[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names_sorted[::-1], imp_sorted[::-1], color=sns.color_palette("crest", len(names)))
    ax.set_xlabel("Relative Importance")
    ax.set_title("Feature Importance (Random Forest)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    core_results = train_core_model()
    enhanced_results = train_enhanced_models()

    all_metrics = {
        "core_model": {
            "features": ["SquareFootage", "Bedrooms", "Bathrooms"],
            "coefficients": core_results["coefficients"],
            "intercept": core_results["intercept"],
            "train_metrics": core_results["train_metrics"],
            "test_metrics": core_results["test_metrics"],
            "cv_r2_mean": core_results["cv_r2_mean"],
            "cv_r2_std": core_results["cv_r2_std"],
            "cv_rmse_mean": core_results["cv_rmse_mean"],
            "cv_rmse_std": core_results["cv_rmse_std"],
            "n_train": core_results["n_train"],
            "n_test": core_results["n_test"],
            "outliers_removed": core_results["outliers_removed"],
        },
        "enhanced_models": enhanced_results,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nAll metrics saved to {METRICS_PATH}")
    print(f"All figures saved to {FIG_DIR}")
    print(f"All models saved to {MODEL_DIR}")
