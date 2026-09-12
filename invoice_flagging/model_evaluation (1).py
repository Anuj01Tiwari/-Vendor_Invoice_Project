"""
invoice_data_evaluation.py
===========================
Complete evaluation and visualisation suite for the Invoice Risk
Flagging model.

Diagnostics produced
--------------------
  1.  Classification report   (Accuracy, Precision, Recall, F1)
  2.  Confusion matrix
  3.  ROC-AUC curve
  4.  Precision-Recall curve
  5.  Feature importance bar chart
  6.  Statistical t-test table  (flagged vs normal, 9 features)
  7.  EDA panel — Freight distribution / Invoice $ vs Freight /
                  Receiving delay by flag
  8.  Label distribution bar chart
  9.  Correlation heatmap

All plots are saved to  evaluation_plots/invoice/

Author : Abhishek Thakur
Project: Invoice Risk Flagging
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
    accuracy_score,
)

from invoice_data_preprocessing import run_preprocessing, FEATURE_COLS, TARGET_COL

# ─────────────────────────── logging ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────── style & dirs ────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

PLOTS_DIR  = Path("evaluation_plots/invoice")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = Path("models")


# ════════════════════════════════════════════════════════════════
# 0.  UTILITIES
# ════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, filename: str) -> None:
    """Save figure to PLOTS_DIR and close it."""
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


def load_model(filename: str = "invoice_flagging_model.pkl"):
    """Load the persisted invoice model from models/."""
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run invoice_train.py first."
        )
    model = joblib.load(path)
    logger.info("Model loaded ← %s", path)
    return model


# ════════════════════════════════════════════════════════════════
# 1.  CLASSIFICATION METRICS SUMMARY
# ════════════════════════════════════════════════════════════════

def classification_summary(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Print full classification report.
    Returns the report as a DataFrame.
    """
    y_pred  = model.predict(X_test)
    report  = classification_report(
        y_test, y_pred,
        target_names=["Normal (0)", "Risky (1)"],
        output_dict=True,
    )
    rep_df = pd.DataFrame(report).T

    logger.info(
        "\n── Invoice Classification Report ──\n%s",
        classification_report(y_test, y_pred,
                               target_names=["Normal (0)", "Risky (1)"]),
    )

    acc  = accuracy_score(y_test, y_pred)
    f1_r = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    logger.info("Overall Accuracy : %.4f", acc)
    logger.info("F1 (Risky class) : %.4f", f1_r)

    return rep_df


# ════════════════════════════════════════════════════════════════
# 2.  CONFUSION MATRIX
# ════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Annotated confusion matrix with class labels."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=["Normal", "Risky"],
        colorbar=False,
        ax=ax,
        cmap="Blues",
    )
    ax.set_title("Invoice Flagging — Confusion Matrix")
    _save(fig, "01_confusion_matrix.png")


# ════════════════════════════════════════════════════════════════
# 3.  ROC CURVE
# ════════════════════════════════════════════════════════════════

def plot_roc_curve(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> float | None:
    """
    ROC-AUC curve.
    Skipped (with warning) if the model has no predict_proba.

    Returns
    -------
    float (AUC score) or None
    """
    if not hasattr(model, "predict_proba"):
        logger.warning("Model has no predict_proba — skipping ROC curve.")
        return None

    y_prob         = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _    = roc_curve(y_test, y_prob)
    roc_auc_score  = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, color="#4c72b0",
            label=f"AUC = {roc_auc_score:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Random Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Invoice Flagging — ROC Curve")
    ax.legend(loc="lower right")
    _save(fig, "02_roc_curve.png")

    logger.info("ROC-AUC: %.4f", roc_auc_score)
    return roc_auc_score


# ════════════════════════════════════════════════════════════════
# 4.  PRECISION-RECALL CURVE
# ════════════════════════════════════════════════════════════════

def plot_precision_recall_curve(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> float | None:
    """
    Precision–Recall curve — especially important for imbalanced classes.
    Skipped if model has no predict_proba.

    Returns
    -------
    float (PR-AUC) or None
    """
    if not hasattr(model, "predict_proba"):
        logger.warning("Model has no predict_proba — skipping PR curve.")
        return None

    y_prob       = model.predict_proba(X_test)[:, 1]
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    pr_auc_score = auc(rec, prec)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, lw=2, color="#c44e52",
            label=f"PR-AUC = {pr_auc_score:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Invoice Flagging — Precision–Recall Curve")
    ax.legend()
    _save(fig, "03_precision_recall_curve.png")

    logger.info("PR-AUC: %.4f", pr_auc_score)
    return pr_auc_score


# ════════════════════════════════════════════════════════════════
# 5.  FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════

def plot_feature_importance(model) -> None:
    """
    Horizontal bar chart of feature importances.
    Skipped if model has no feature_importances_ attribute.
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model has no feature_importances_ — skipping.")
        return

    feat_df = pd.DataFrame({
        "feature"   : FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feat_df["feature"], feat_df["importance"],
            color="#4c72b0", edgecolor="white")
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance — Invoice Flagging Model")
    _save(fig, "04_feature_importance.png")


# ════════════════════════════════════════════════════════════════
# 6.  STATISTICAL T-TEST ANALYSIS
# ════════════════════════════════════════════════════════════════

def run_ttest_analysis(invoice_df: pd.DataFrame) -> pd.DataFrame:
    """
    Welch two-sample t-test for each feature:
    Are flagged invoices statistically different from normal ones?

    Returns
    -------
    pd.DataFrame — one row per feature with means, t-stat, p-value,
                   and significance flag.
    """
    logger.info("Running Welch t-tests (flagged vs normal) …")

    flagged = invoice_df[invoice_df[TARGET_COL] == 1]
    normal  = invoice_df[invoice_df[TARGET_COL] == 0]
    results = []

    for col in FEATURE_COLS:
        if col not in invoice_df.columns:
            continue
        f_mean = flagged[col].mean()
        n_mean = normal[col].mean()
        t_stat, p_val = ttest_ind(
            flagged[col].dropna(),
            normal[col].dropna(),
            equal_var=False,
        )
        results.append({
            "Feature"      : col,
            "Flagged_Mean" : round(f_mean, 2),
            "Normal_Mean"  : round(n_mean, 2),
            "T_Stat"       : round(t_stat, 4),
            "P_Value"      : round(p_val,  4),
            "Significant"  : p_val < 0.05,
        })

    results_df = pd.DataFrame(results).set_index("Feature")
    sig_count  = results_df["Significant"].sum()
    logger.info(
        "\n── T-Test Results  (%d/%d features significant, p < 0.05) ──\n%s",
        sig_count, len(results_df), results_df.to_string(),
    )
    return results_df


# ════════════════════════════════════════════════════════════════
# 7.  EDA PANEL
# ════════════════════════════════════════════════════════════════

def plot_eda_panel(invoice_df: pd.DataFrame) -> None:
    """
    Three-panel EDA:
      Left   — Freight distribution histogram
      Centre — Invoice Dollars vs Freight  (colour = flag)
      Right  — Avg Receiving Delay by risk flag  (box plot)
    """
    palette = {0: "#4c72b0", 1: "#c44e52"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Panel 1 — Freight distribution
    sns.histplot(invoice_df["Freight"], bins=40,
                 ax=axes[0], color="#4c72b0", edgecolor="white")
    axes[0].set_title("Freight Distribution")
    axes[0].set_xlabel("Freight ($)")

    # Panel 2 — Invoice $ vs Freight coloured by flag
    for flag, grp in invoice_df.groupby(TARGET_COL):
        axes[1].scatter(
            grp["invoice_dollars"], grp["Freight"],
            alpha=0.4, s=18, edgecolors="none",
            color=palette[flag],
            label="Risky" if flag else "Normal",
        )
    axes[1].set_xlabel("Invoice Dollars ($)")
    axes[1].set_ylabel("Freight ($)")
    axes[1].set_title("Invoice $ vs Freight")
    axes[1].legend()

    # Panel 3 — Receiving delay by flag
    sns.boxplot(
        x=TARGET_COL, y="avg_receiving_delay",
        data=invoice_df, ax=axes[2],
        palette=palette, order=[0, 1],
    )
    axes[2].set_xticklabels(["Normal (0)", "Risky (1)"])
    axes[2].set_xlabel("Invoice Risk Flag")
    axes[2].set_ylabel("Avg Receiving Delay (days)")
    axes[2].set_title("Receiving Delay vs Risk Flag")

    fig.tight_layout()
    _save(fig, "05_eda_panel.png")


# ════════════════════════════════════════════════════════════════
# 8.  LABEL DISTRIBUTION
# ════════════════════════════════════════════════════════════════

def plot_label_distribution(invoice_df: pd.DataFrame) -> None:
    """Bar chart showing count of normal vs risky invoices."""
    counts = invoice_df[TARGET_COL].value_counts().rename(
        index={0: "Normal", 1: "Risky"}
    )
    fig, ax = plt.subplots(figsize=(4, 3))
    counts.plot(
        kind="bar", ax=ax, rot=0,
        color=["#4c72b0", "#c44e52"], edgecolor="white",
    )
    ax.set_xlabel("Invoice Category")
    ax.set_ylabel("Count")
    ax.set_title("Risk Label Distribution")
    for i, v in enumerate(counts):
        ax.text(i, v + counts.max() * 0.01, str(v), ha="center", fontsize=10)
    _save(fig, "06_label_distribution.png")


# ════════════════════════════════════════════════════════════════
# 9.  CORRELATION HEATMAP
# ════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(invoice_df: pd.DataFrame) -> None:
    """Full correlation heatmap across all feature columns."""
    numeric_cols = [c for c in FEATURE_COLS if c in invoice_df.columns]
    corr = invoice_df[numeric_cols + [TARGET_COL]].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, linewidths=0.4, ax=ax,
        vmin=-1, vmax=1,
    )
    ax.set_title("Feature Correlation Matrix — Invoice Dataset")
    _save(fig, "07_correlation_heatmap.png")


# ════════════════════════════════════════════════════════════════
# 10.  FULL EVALUATION PIPELINE
# ════════════════════════════════════════════════════════════════

def run_evaluation(db_path: str = "inventory.db") -> dict:
    """
    Load data, load model, execute all diagnostics.

    Returns
    -------
    dict with keys:
        report_df, ttest_df, roc_auc, pr_auc, model, processed_df
    """
    logger.info("━" * 60)
    logger.info("INVOICE RISK FLAGGING — EVALUATION")
    logger.info("━" * 60)

    # Data
    data        = run_preprocessing(db_path)
    X_test      = data["X_test"]
    y_test      = data["y_test"]
    processed   = data["processed_df"]

    # Model
    model = load_model()

    # ── Metrics ───────────────────────────────────────────────
    report_df = classification_summary(model, X_test, y_test)

    # ── Plots ─────────────────────────────────────────────────
    plot_confusion_matrix(model, X_test, y_test)
    roc_auc = plot_roc_curve(model, X_test, y_test)
    pr_auc  = plot_precision_recall_curve(model, X_test, y_test)
    plot_feature_importance(model)
    plot_eda_panel(processed)
    plot_label_distribution(processed)
    plot_correlation_heatmap(processed)

    # ── T-Test ────────────────────────────────────────────────
    ttest_df = run_ttest_analysis(processed)

    data["conn"].close()
    logger.info(
        "Evaluation complete. All plots saved to: %s",
        PLOTS_DIR.resolve(),
    )

    return {
        "report_df"   : report_df,
        "ttest_df"    : ttest_df,
        "roc_auc"     : roc_auc,
        "pr_auc"      : pr_auc,
        "model"       : model,
        "processed_df": processed,
    }


# ─── entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    result = run_evaluation()
    print("\n── Classification Report ──")
    print(result["report_df"])
    print("\n── T-Test Results (significant features only) ──")
    print(result["ttest_df"][result["ttest_df"]["Significant"]])