"""
freight_data_evaluation.py
==========================
Complete evaluation and visualisation suite for the Freight Cost
Prediction model.

Diagnostics produced
--------------------
  1. Regression metrics  (R², MAE, RMSE, MAPE)
  2. Actual vs Predicted scatter
  3. Residual diagnostics  (residuals vs fitted + distribution)
  4. Freight-per-unit economies-of-scale bar chart
  5. Correlation heatmap  (Quantity / Dollars / Freight)
  6. Model comparison table

All plots are saved to  evaluation_plots/freight/

Author : Data Science Team
Project: Predicting Freight Cost
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from freight_data_preprocessing import run_preprocessing

# ─────────────────────────── logging ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────── style & dirs ────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

PLOTS_DIR  = Path("evaluation_plots/freight")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = Path("models")


# ════════════════════════════════════════════════════════════════
# 0.  HELPER
# ════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, filename: str) -> None:
    """Save figure and close it."""
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


def load_model(filename: str = "predicting_freight_model.pkl"):
    """Load the persisted freight model from models/."""
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run freight_train.py first."
        )
    model = joblib.load(path)
    logger.info("Model loaded ← %s", path)
    return model


# ════════════════════════════════════════════════════════════════
# 1.  REGRESSION METRICS
# ════════════════════════════════════════════════════════════════

def regression_metrics(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Compute and log R², MAE, RMSE, and MAPE.

    Returns
    -------
    pd.DataFrame  — one row per metric
    """
    y_pred = model.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(
        np.abs((y_test.values - y_pred) / np.where(y_test.values == 0, np.nan, y_test.values))
    ) * 100

    metrics = {
        "R²"  : round(r2,   4),
        "MAE" : round(mae,  4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 2),
    }
    summary = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
    logger.info("\n── Freight Regression Metrics ──\n%s", summary.to_string())
    return summary


# ════════════════════════════════════════════════════════════════
# 2.  CORRELATION HEATMAP
# ════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(vendor_df: pd.DataFrame) -> None:
    """
    Heatmap showing pairwise correlation between Quantity,
    Dollars, and Freight — replicates notebook EDA cell.
    """
    corr = vendor_df[["Quantity", "Dollars", "Freight"]].corr()

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues",
                linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Feature Correlation  (Quantity / Dollars / Freight)")
    _save(fig, "01_correlation_heatmap.png")


# ════════════════════════════════════════════════════════════════
# 3.  SCATTER: QUANTITY vs FREIGHT  &  DOLLARS vs FREIGHT
# ════════════════════════════════════════════════════════════════

def plot_feature_vs_freight(vendor_df: pd.DataFrame) -> None:
    """
    Side-by-side scatter: how Quantity and Dollars individually
    relate to Freight — mirrors the notebook scatter plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].scatter(vendor_df["Quantity"], vendor_df["Freight"],
                    color="#f57a55", alpha=0.4, s=20, edgecolors="none")
    axes[0].set_xlabel("Quantity")
    axes[0].set_ylabel("Freight Cost ($)")
    axes[0].set_title("Quantity vs Freight")

    axes[1].scatter(vendor_df["Dollars"], vendor_df["Freight"],
                    color="#7f1e5a", alpha=0.4, s=20, edgecolors="none")
    axes[1].set_xlabel("Order Value (Dollars)")
    axes[1].set_ylabel("Freight Cost ($)")
    axes[1].set_title("Dollars vs Freight")

    fig.tight_layout()
    _save(fig, "02_features_vs_freight.png")


# ════════════════════════════════════════════════════════════════
# 4.  ACTUAL vs PREDICTED SCATTER
# ════════════════════════════════════════════════════════════════

def plot_actual_vs_predicted(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Scatter of actual freight vs predicted freight values,
    with an ideal y = x identity line.
    """
    y_pred = model.predict(X_test)

    # Sort by Dollars for a clean line plot overlay
    sort_idx = X_test["Dollars"].argsort()
    x_sorted = X_test["Dollars"].iloc[sort_idx]
    y_pred_sorted = y_pred[sort_idx]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X_test["Dollars"], y_test,
               color="blue", alpha=0.45, s=18, label="Actual Freight")
    ax.plot(x_sorted, y_pred_sorted,
            color="red", linewidth=1.8, label="Predicted Freight")
    ax.set_xlabel("Order Value (Dollars)")
    ax.set_ylabel("Freight Cost ($)")
    ax.set_title("Actual vs Predicted Freight Cost")
    ax.legend()
    _save(fig, "03_actual_vs_predicted.png")


# ════════════════════════════════════════════════════════════════
# 5.  RESIDUAL DIAGNOSTICS
# ════════════════════════════════════════════════════════════════

def plot_residuals(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Two-panel residual diagnostic:
      Left  — Residuals vs Fitted (checks homoscedasticity)
      Right — Residual histogram  (checks normality)
    """
    y_pred    = model.predict(X_test)
    residuals = y_test.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Residuals vs fitted
    axes[0].scatter(y_pred, residuals, alpha=0.4, color="#dd8452",
                    s=20, edgecolors="none")
    axes[0].axhline(0, color="red", linewidth=1.3, linestyle="--")
    axes[0].set_xlabel("Fitted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Fitted")

    # Residual distribution
    axes[1].hist(residuals, bins=45, color="#55a868", edgecolor="white")
    axes[1].axvline(0, color="red", linewidth=1.3, linestyle="--")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    fig.tight_layout()
    _save(fig, "04_residual_diagnostics.png")


# ════════════════════════════════════════════════════════════════
# 6.  FREIGHT-PER-UNIT ECONOMIES OF SCALE
# ════════════════════════════════════════════════════════════════

def plot_freight_per_unit(vendor_df: pd.DataFrame) -> None:
    """
    Bar chart: average freight-per-unit for low-quantity orders
    vs high-quantity orders.

    Business insight: bulk orders reduce freight cost per unit
    — economies of scale in shipping.
    """
    df = vendor_df.copy()
    if "freight_per_unit" not in df.columns:
        df["freight_per_unit"] = df["Freight"] / df["Quantity"].replace(0, np.nan)

    q25 = df["Quantity"].quantile(0.25)
    q75 = df["Quantity"].quantile(0.75)

    low_fpu  = df.loc[df["Quantity"] <= q25, "freight_per_unit"].mean()
    high_fpu = df.loc[df["Quantity"] >= q75, "freight_per_unit"].mean()

    logger.info(
        "Freight/unit  |  Low-qty: $%.4f  |  High-qty: $%.4f",
        low_fpu, high_fpu,
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["Low Quantity\n(≤ Q25)", "High Quantity\n(≥ Q75)"],
        [low_fpu, high_fpu],
        color=["#c44e52", "#4c72b0"],
        width=0.4, edgecolor="white",
    )
    for bar, val in zip(bars, [low_fpu, high_fpu]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + low_fpu * 0.02,
            f"${val:.2f}", ha="center", fontsize=11, fontweight="bold",
        )
    ax.set_ylabel("Avg Freight per Unit ($)")
    ax.set_title("Economies of Scale in Shipping")
    _save(fig, "05_freight_per_unit.png")


# ════════════════════════════════════════════════════════════════
# 7.  FULL EVALUATION PIPELINE
# ════════════════════════════════════════════════════════════════

def run_evaluation(db_path: str = "inventory.db") -> dict:
    """
    Load data, load model, run all diagnostics.

    Returns
    -------
    dict with keys: metrics_df, model, processed_df
    """
    logger.info("━" * 55)
    logger.info("FREIGHT COST PREDICTION — EVALUATION")
    logger.info("━" * 55)

    # Data
    data        = run_preprocessing(db_path)
    X_test      = data["X_test"]
    y_test      = data["y_test"]
    processed   = data["processed_df"]

    # Model
    model = load_model()

    # Metrics
    metrics_df = regression_metrics(model, X_test, y_test)

    # Plots
    plot_correlation_heatmap(processed)
    plot_feature_vs_freight(processed)
    plot_actual_vs_predicted(model, X_test, y_test)
    plot_residuals(model, X_test, y_test)
    plot_freight_per_unit(processed)

    data["conn"].close()
    logger.info(
        "Evaluation complete. All plots saved to: %s",
        PLOTS_DIR.resolve(),
    )
    return {
        "metrics_df"  : metrics_df,
        "model"       : model,
        "processed_df": processed,
    }


# ─── entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    result = run_evaluation()
    print("\n── Freight Evaluation Metrics ──")
    print(result["metrics_df"])