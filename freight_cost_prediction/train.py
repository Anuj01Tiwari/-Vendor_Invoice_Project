"""
freight_train.py
================
Trains and persists regression models for the Freight Cost Prediction project.

Models compared
---------------
  1. Linear Regression
  2. Decision Tree Regressor
  3. Random Forest Regressor

Selection criterion : highest R²
Saved artefact      : models/predicting_freight_model.pkl

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

from sklearn.linear_model  import LinearRegression
from sklearn.tree           import DecisionTreeRegressor
from sklearn.ensemble       import RandomForestRegressor
from sklearn.metrics        import r2_score, mean_absolute_error, mean_squared_error

from freight_data_preprocessing import run_preprocessing

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
# 1.  MODEL EVALUATION HELPER
# ════════════════════════════════════════════════════════════════

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    name: str,
) -> dict:
    """
    Predict on X_test and return R², MAE, RMSE metrics.

    Parameters
    ----------
    model  : fitted sklearn estimator
    X_test : test features
    y_test : true target values
    name   : display name for logging

    Returns
    -------
    dict with keys: model, R2, MAE, RMSE
    """
    y_pred = model.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    logger.info(
        "%-30s  R²=%.4f  |  MAE=%.4f  |  RMSE=%.4f",
        name, r2, mae, rmse,
    )
    return {"model": name, "R2": round(r2, 4), "MAE": round(mae, 4), "RMSE": round(rmse, 4)}


# ════════════════════════════════════════════════════════════════
# 2.  MODEL DEFINITIONS
# ════════════════════════════════════════════════════════════════

def get_candidate_models() -> dict:
    """
    Return a dict of { model_name: unfitted_estimator }.
    Hyperparameters match the notebook defaults.
    """
    return {
        "LinearRegression"     : LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=5, random_state=42),
        "RandomForestRegressor": RandomForestRegressor(max_depth=6, random_state=42, n_jobs=-1),
    }


# ════════════════════════════════════════════════════════════════
# 3.  TRAINING
# ════════════════════════════════════════════════════════════════

def train_all_models(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_test:  pd.Series,
) -> tuple[dict, pd.DataFrame]:
    """
    Fit all candidate models, evaluate each, and return results.

    Returns
    -------
    trained_models : dict { model_name: fitted_estimator }
    results_df     : DataFrame with R², MAE, RMSE per model
    """
    logger.info("=" * 55)
    logger.info("FREIGHT COST PREDICTION — MODEL TRAINING")
    logger.info("=" * 55)
    logger.info(
        "Train size: %d  |  Test size: %d  |  Features: %s",
        len(X_train), len(X_test), list(X_train.columns),
    )

    candidates     = get_candidate_models()
    trained_models = {}
    results        = []

    for name, model in candidates.items():
        logger.info("Training  →  %s …", name)
        model.fit(X_train, y_train)
        trained_models[name] = model
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)

    results_df = pd.DataFrame(results).set_index("model")
    logger.info("\n── Model Comparison ──\n%s", results_df.to_string())
    return trained_models, results_df


# ════════════════════════════════════════════════════════════════
# 4.  MODEL SELECTION
# ════════════════════════════════════════════════════════════════

def select_best_model(
    trained_models: dict,
    results_df: pd.DataFrame,
    criterion: str = "R2",
) -> tuple[object, str]:
    """
    Pick the model with the highest value of `criterion`.

    Parameters
    ----------
    trained_models : dict returned by train_all_models
    results_df     : DataFrame with metric columns
    criterion      : column name to maximise (default "R2")

    Returns
    -------
    best_model, best_name
    """
    best_name  = results_df[criterion].idxmax()
    best_model = trained_models[best_name]
    best_score = results_df.loc[best_name, criterion]

    logger.info(
        "Best model selected  →  %s  (R²=%.4f)",
        best_name, best_score,
    )
    return best_model, best_name


# ════════════════════════════════════════════════════════════════
# 5.  VISUALISATION
# ════════════════════════════════════════════════════════════════

def plot_actual_vs_predicted(
    best_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: str = "freight_actual_vs_predicted.png",
) -> None:
    """
    Scatter plot of actual vs predicted freight cost with the
    Linear Regression line overlaid.
    """
    y_pred = best_model.predict(X_test)

    plt.figure(figsize=(6, 5))
    plt.scatter(
        X_test["Dollars"], y_test,
        color="blue", alpha=0.5, label="Actual Freight", s=20,
    )
    plt.plot(
        X_test["Dollars"],
        y_pred,
        color="red", linewidth=1.5, label="Predicted Freight",
    )
    plt.xlabel("Order Value (Dollars)")
    plt.ylabel("Freight Cost")
    plt.title("Actual vs Predicted Freight Cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Plot saved → %s", save_path)


def plot_model_comparison(
    results_df: pd.DataFrame,
    save_path: str = "freight_model_comparison.png",
) -> None:
    """Bar chart comparing R² across all trained models."""
    fig, ax = plt.subplots(figsize=(7, 4))
    results_df["R2"].sort_values().plot(
        kind="barh", ax=ax, color="#4c72b0", edgecolor="white"
    )
    ax.set_xlabel("R² Score")
    ax.set_title("Freight Model Comparison — R²")
    ax.axvline(x=0.9, color="red", linestyle="--", linewidth=1, label="R²=0.9 threshold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Plot saved → %s", save_path)


# ════════════════════════════════════════════════════════════════
# 6.  INFERENCE HELPER
# ════════════════════════════════════════════════════════════════

def predict_freight_cost(model, quantity: float, dollars: float) -> float:
    """
    Predict freight cost for a single order.

    Parameters
    ----------
    model    : trained sklearn estimator
    quantity : number of units ordered
    dollars  : total order value in USD

    Returns
    -------
    float — predicted freight cost
    """
    prediction = model.predict([[quantity, dollars]])
    result = round(float(prediction[0]), 4)
    logger.info(
        "Prediction  →  Qty=%s  Dollars=$%s  →  Freight=$%.4f",
        quantity, dollars, result,
    )
    return result


# ════════════════════════════════════════════════════════════════
# 7.  SAVE MODEL
# ════════════════════════════════════════════════════════════════

def save_model(model, filename: str = "predicting_freight_model.pkl") -> Path:
    """Persist model to models/ directory using joblib."""
    save_path = MODELS_DIR / filename
    joblib.dump(model, save_path)
    logger.info("Model saved → %s", save_path.resolve())
    return save_path


# ════════════════════════════════════════════════════════════════
# 8.  FULL TRAINING PIPELINE
# ════════════════════════════════════════════════════════════════

def run_training(db_path: str = "inventory.db") -> dict:
    """
    Execute the complete training pipeline end-to-end.

    Steps
    -----
    1. Preprocess data
    2. Train all candidate models
    3. Evaluate and select best
    4. Plot results
    5. Run a sample prediction
    6. Save model

    Returns
    -------
    dict with keys: best_model, best_name, results_df
    """
    # Step 1 – preprocess
    data = run_preprocessing(db_path)

    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]

    # Step 2 & 3 – train and select
    trained_models, results_df = train_all_models(X_train, X_test, y_train, y_test)
    best_model, best_name      = select_best_model(trained_models, results_df)

    # Step 4 – plots
    plot_actual_vs_predicted(best_model, X_test, y_test)
    plot_model_comparison(results_df)

    # Step 5 – sample prediction
    sample = predict_freight_cost(best_model, quantity=500, dollars=12_000)
    logger.info("Sample freight prediction (qty=500, $12k): $%.4f", sample)

    # Step 6 – save
    save_model(best_model)

    data["conn"].close()
    logger.info("Training pipeline complete.")

    return {
        "best_model" : best_model,
        "best_name"  : best_name,
        "results_df" : results_df,
    }


# ─── entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    output = run_training()
    print("\n── Final Model Comparison ──")
    print(output["results_df"])
    print(f"\nBest Model: {output['best_name']}")