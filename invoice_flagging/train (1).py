"""
invoice_train.py
================
Trains and persists classification models for the Invoice Risk
Flagging project.

Models compared
---------------
  1. Logistic Regression      (Pipeline: StandardScaler → LR)
  2. Decision Tree Classifier (class_weight='balanced')
  3. Random Forest Classifier (class_weight='balanced')

Tuning
------
  GridSearchCV on DecisionTreeClassifier (cv=5, scoring=f1)
  → best estimator becomes the final model.

Design notes
------------
  • All models are wrapped in sklearn Pipeline objects so that
    preprocessing (scaling) is bundled with the estimator.
    This prevents data leakage during cross-validation and
    makes deployment safe — one object handles both steps.
  • class_weight='balanced' compensates for class imbalance
    without requiring SMOTE or oversampling.
  • Leaky features (invoice_dollars, total_item_dollars,
    avg_receiving_delay) are excluded from FEATURE_COLS —
    they were used to CREATE the labels and cannot be inputs.

Saved artefact : models/invoice_flagging_model.pkl

Author : Abhishek Thakur
Project: Invoice Risk Flagging
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model    import LogisticRegression
from sklearn.tree             import DecisionTreeClassifier
from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import GridSearchCV
from sklearn.pipeline         import Pipeline
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import (
    classification_report,
    f1_score,
    accuracy_score,
)

from invoice_data_preprocessing import run_preprocessing, FEATURE_COLS

# ─────────────────────────── logging ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════════
# 1.  EVALUATION HELPER
# ════════════════════════════════════════════════════════════════

def evaluate_classifier(
    model,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    name:    str,
) -> dict:
    """
    Predict on X_test, log classification report, return summary metrics.

    Returns
    -------
    dict with keys: model, Accuracy, F1_weighted, F1_risky
    """
    y_pred   = model.predict(X_test)
    acc      = accuracy_score(y_test, y_pred)
    f1_w     = f1_score(y_test, y_pred, average="weighted")
    f1_risky = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    logger.info(
        "%-35s  Acc=%.4f  F1(weighted)=%.4f  F1(risky)=%.4f",
        name, acc, f1_w, f1_risky,
    )
    logger.info("\n%s", classification_report(y_test, y_pred,
                target_names=["Normal", "Risky"]))

    return {
        "model"       : name,
        "Accuracy"    : round(acc, 4),
        "F1_weighted" : round(f1_w, 4),
        "F1_risky"    : round(f1_risky, 4),
    }


# ════════════════════════════════════════════════════════════════
# 2.  BASELINE TRAINING
# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════
# 2.  BASELINE TRAINING
# ════════════════════════════════════════════════════════════════

def train_baseline_models(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_test:  pd.Series,
) -> tuple[dict, pd.DataFrame]:
    """
    Train three baseline Pipelines (scaler + classifier).

    Each model is a sklearn Pipeline so that:
      • Scaling is fitted only on training data (no leakage in CV)
      • The saved .pkl contains both scaler and classifier
      • Inference requires only model.predict(X_raw)

    class_weight='balanced' is applied to all classifiers to
    compensate for label imbalance.

    Returns
    -------
    trained_models : dict { name: fitted Pipeline }
    results_df     : DataFrame with metrics per model
    """
    logger.info("=" * 60)
    logger.info("INVOICE RISK FLAGGING — BASELINE TRAINING")
    logger.info("=" * 60)

    candidates = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            )),
        ]),
        "DecisionTreeClassifier": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    DecisionTreeClassifier(
                max_depth=5, class_weight="balanced", random_state=42
            )),
        ]),
        "RandomForestClassifier": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1
            )),
        ]),
    }

    trained_models = {}
    results        = []

    for name, pipeline in candidates.items():
        logger.info("Training  →  %s …", name)
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        metrics = evaluate_classifier(pipeline, X_test, y_test, name)
        results.append(metrics)

    results_df = pd.DataFrame(results).set_index("model")
    logger.info("\n── Baseline Comparison ──\n%s", results_df.to_string())
    return trained_models, results_df


# ════════════════════════════════════════════════════════════════
# 3.  HYPERPARAMETER TUNING  (Decision Tree)
# ════════════════════════════════════════════════════════════════

PARAM_GRID = {
    "clf__max_depth"        : [3, 5, 7, 10, None],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf" : [1, 2, 4],
}


def tune_decision_tree(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_test:  pd.Series,
) -> tuple[object, dict, dict]:
    """
    GridSearchCV on a Pipeline(StandardScaler → DecisionTreeClassifier).
    Uses Pipeline param syntax: clf__<param>.

    Returns
    -------
    best_pipeline, best_params, tuned_metrics
    """
    logger.info("=" * 60)
    logger.info("HYPERPARAMETER TUNING — Decision Tree  (GridSearchCV)")
    logger.info("=" * 60)
    logger.info("Param grid: %s", PARAM_GRID)

    base_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    DecisionTreeClassifier(
            class_weight="balanced", random_state=42
        )),
    ])

    grid_search = GridSearchCV(
        estimator  = base_pipeline,
        param_grid = PARAM_GRID,
        cv         = 5,
        scoring    = "f1",
        n_jobs     = -1,
        verbose    = 1,
        refit      = True,
    )
    grid_search.fit(X_train, y_train)

    logger.info("Best params : %s", grid_search.best_params_)
    logger.info("Best CV F1  : %.4f", grid_search.best_score_)

    best_pipeline  = grid_search.best_estimator_
    tuned_metrics  = evaluate_classifier(
        best_pipeline, X_test, y_test, "TunedDecisionTree"
    )
    tuned_metrics["best_params"] = str(grid_search.best_params_)

    return best_pipeline, grid_search.best_params_, tuned_metrics


# ════════════════════════════════════════════════════════════════
# 4.  FEATURE IMPORTANCE PLOT
# ════════════════════════════════════════════════════════════════

def plot_feature_importance(
    model,
    feature_names: list[str],
    title: str = "Feature Importance — Invoice Flagging Model",
    save_path: str = "invoice_feature_importance.png",
) -> pd.DataFrame:
    """
    Horizontal bar chart of feature importances for tree models.

    Returns
    -------
    pd.DataFrame  — sorted feature importances
    """
    feat_df = pd.DataFrame({
        "feature"   : feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    logger.info("\nFeature Importance:\n%s", feat_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x="importance", y="feature",
        data=feat_df.sort_values("importance"),
        ax=ax, color="#4c72b0", edgecolor="white",
    )
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Plot saved → %s", save_path)

    return feat_df


# ════════════════════════════════════════════════════════════════
# 5.  MODEL COMPARISON PLOT
# ════════════════════════════════════════════════════════════════

def plot_model_comparison(
    results_df: pd.DataFrame,
    save_path: str = "invoice_model_comparison.png",
) -> None:
    """Grouped bar chart comparing Accuracy, F1(weighted), F1(risky)."""
    plot_df = results_df[["Accuracy", "F1_weighted", "F1_risky"]].copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df.plot(kind="bar", ax=ax, rot=20,
                 color=["#4c72b0", "#55a868", "#c44e52"],
                 edgecolor="white")
    ax.set_ylabel("Score")
    ax.set_title("Invoice Risk Flagging — Model Comparison")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Plot saved → %s", save_path)


# ════════════════════════════════════════════════════════════════
# 6.  INFERENCE HELPER
# ════════════════════════════════════════════════════════════════

def predict_invoice_risk(model, invoice_vector: list) -> dict:
    """
    Predict risk flag for a single invoice.

    Parameters
    ----------
    model          : trained sklearn Pipeline
    invoice_vector : list of 6 feature values in order matching
                     FEATURE_COLS:
                     [invoice_quantity, Freight,
                      days_po_to_invoice, days_to_pay,
                      total_brands, total_item_quantity]

    Returns
    -------
    dict with keys: prediction (int), label (str), proba (float or None)
    """
    pred  = int(model.predict([invoice_vector])[0])
    label = "RISKY" if pred == 1 else "NORMAL"

    proba = None
    if hasattr(model, "predict_proba"):
        proba = round(float(model.predict_proba([invoice_vector])[0][1]), 4)

    logger.info(
        "Invoice Prediction  →  %s  (risky probability: %s)", label, proba
    )
    return {"prediction": pred, "label": label, "risk_proba": proba}


# ════════════════════════════════════════════════════════════════
# 7.  SAVE MODEL
# ════════════════════════════════════════════════════════════════

def save_model(model, filename: str = "invoice_flagging_model.pkl") -> Path:
    """Persist model to models/ directory."""
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    logger.info("Model saved → %s", path.resolve())
    return path


# ════════════════════════════════════════════════════════════════
# 8.  FULL TRAINING PIPELINE
# ════════════════════════════════════════════════════════════════

def run_training(db_path: str = "inventory.db") -> dict:
    """
    End-to-end training pipeline:
      1. Preprocess data
      2. Train baseline models
      3. Tune Decision Tree with GridSearchCV
      4. Plot results
      5. Run sample prediction
      6. Save best model

    Returns
    -------
    dict with keys: best_model, best_params, results_df, feat_importance_df
    """
    # Step 1 — preprocess
    data = run_preprocessing(db_path)

    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]

    # Step 2 — baseline training (Pipelines handle scaling internally)
    baseline_models, results_df = train_baseline_models(
        X_train, X_test, y_train, y_test
    )

    # Step 3 — hyperparameter tuning
    best_model, best_params, tuned_metrics = tune_decision_tree(
        X_train, X_test, y_train, y_test
    )
    tuned_row = pd.DataFrame([tuned_metrics]).set_index("model")
    results_df = pd.concat([results_df, tuned_row])

    # Step 4 — plots
    feat_df = plot_feature_importance(best_model.named_steps["clf"], FEATURE_COLS)
    plot_model_comparison(results_df[["Accuracy", "F1_weighted", "F1_risky"]])

    # Step 5 — sample prediction
    # [invoice_quantity, Freight, days_po_to_invoice,
    #  days_to_pay, total_brands, total_item_quantity]
    test_invoice = [20, 120, 5, 10, 3, 200]
    prediction   = predict_invoice_risk(best_model, test_invoice)
    logger.info("Sample invoice result: %s", prediction)

    # Step 6 — save
    save_model(best_model)

    data["conn"].close()
    logger.info("Training pipeline complete.")

    return {
        "best_model"         : best_model,
        "best_params"        : best_params,
        "results_df"         : results_df,
        "feat_importance_df" : feat_df,
    }


# ─── entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    output = run_training()
    print("\n── Final Model Comparison ──")
    print(output["results_df"][["Accuracy", "F1_weighted", "F1_risky"]])
    print(f"\nBest Params: {output['best_params']}")