"""
freight_data_preprocessing.py
==============================
Data ingestion, cleaning, feature engineering, and train-test splitting
specifically for the Freight Cost Prediction project.

Source table : vendor_invoice  (SQLite → inventory.db)
Target column: Freight
Feature cols : Quantity, Dollars

Author : Abhishek Thakur
Project: Predicting Freight Cost
"""

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ─────────────────────────── logging ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# 1.  DATABASE CONNECTION
# ════════════════════════════════════════════════════════════════

def get_connection(db_path: str = "inventory.db") -> sqlite3.Connection:
    """
    Open and return a SQLite connection.

    Parameters
    ----------
    db_path : str  — path to the .db file

    Returns
    -------
    sqlite3.Connection
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found at: {path.resolve()}")

    conn = sqlite3.connect(path)
    logger.info("Connected to database: %s", path.resolve())
    return conn


# ════════════════════════════════════════════════════════════════
# 2.  RAW DATA LOADING
# ════════════════════════════════════════════════════════════════

def load_vendor_invoice(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load the full vendor_invoice table from SQLite.

    Key columns
    -----------
    VendorNumber, PONumber, Quantity, Dollars, Freight,
    InvoiceDate, PODate, PayDate
    """
    query = "SELECT * FROM vendor_invoice"
    df = pd.read_sql(query, conn)
    logger.info("Raw vendor_invoice loaded  →  %d rows × %d cols", *df.shape)
    return df


def preview_data(df: pd.DataFrame) -> None:
    """Log a quick snapshot of the raw DataFrame."""
    logger.info("\n%s", df.head())
    logger.info("\nDtypes:\n%s", df.dtypes)
    logger.info("\nDescriptive stats:\n%s", df.describe().round(2))


# ════════════════════════════════════════════════════════════════
# 3.  DATA CLEANING
# ════════════════════════════════════════════════════════════════

def report_missing(df: pd.DataFrame) -> pd.Series:
    """Log and return missing-value counts per column."""
    missing = df.isnull().sum()
    pct     = (missing / len(df) * 100).round(2)
    report  = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    has_missing = report[report["missing_count"] > 0]
    if has_missing.empty:
        logger.info("No missing values found.")
    else:
        logger.warning("Missing values detected:\n%s", has_missing)
    return missing


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows and log the count removed."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        logger.warning("Removed %d duplicate rows.", removed)
    else:
        logger.info("No duplicate rows found.")
    return df


def filter_positive_freight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only records where Freight > 0.
    Zero-freight records indicate free shipping or data errors
    and would skew the regression.
    """
    before = len(df)
    df = df[df["Freight"] > 0].copy()
    logger.info("Freight > 0 filter: %d → %d rows (removed %d)",
                before, len(df), before - len(df))
    return df


def drop_nulls_in_required_cols(
    df: pd.DataFrame,
    required_cols: list[str],
) -> pd.DataFrame:
    """Drop rows with NaN in any of the required columns."""
    before = len(df)
    df = df.dropna(subset=required_cols)
    logger.info(
        "dropna on %s: %d → %d rows", required_cols, before, len(df)
    )
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
      1. Remove duplicates
      2. Report missing values
      3. Filter freight > 0
      4. Drop nulls in feature / target columns
    """
    df = remove_duplicates(df)
    report_missing(df)
    df = filter_positive_freight(df)
    df = drop_nulls_in_required_cols(df, ["Quantity", "Dollars", "Freight"])
    logger.info("Cleaning complete  →  %d rows remain.", len(df))
    return df


# ════════════════════════════════════════════════════════════════
# 4.  FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived columns for analysis and modelling.

    New columns
    -----------
    freight_per_unit : Freight / Quantity  — logistics efficiency metric
    order_size       : low / medium / high — based on Q25 / Q75 of Quantity
    """
    df = df.copy()

    # Freight cost per unit (useful for EDA; not a model input)
    df["freight_per_unit"] = (
        df["Freight"] / df["Quantity"].replace(0, np.nan)
    )

    # Order-size bucket
    q25 = df["Quantity"].quantile(0.25)
    q75 = df["Quantity"].quantile(0.75)
    df["order_size"] = pd.cut(
        df["Quantity"],
        bins=[-np.inf, q25, q75, np.inf],
        labels=["low", "medium", "high"],
    )

    # Log insights
    low_fpu  = df.loc[df["Quantity"] <= q25, "freight_per_unit"].mean()
    high_fpu = df.loc[df["Quantity"] >= q75, "freight_per_unit"].mean()
    logger.info(
        "Freight/unit  |  Low-qty: $%.4f  |  High-qty: $%.4f  "
        "(bulk discount = %.1f%%)",
        low_fpu, high_fpu,
        (low_fpu - high_fpu) / low_fpu * 100 if low_fpu else 0,
    )
    logger.info("Order-size distribution:\n%s", df["order_size"].value_counts())
    return df


# ════════════════════════════════════════════════════════════════
# 5.  TRAIN-TEST SPLIT
# ════════════════════════════════════════════════════════════════

FEATURE_COLS = ["Quantity", "Dollars"]
TARGET_COL   = "Freight"


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split cleaned DataFrame into train and test sets.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(
        "Train-test split (%.0f/%.0f)  →  train: %d  |  test: %d",
        (1 - test_size) * 100, test_size * 100,
        len(X_train), len(X_test),
    )
    return X_train, X_test, y_train, y_test


# ════════════════════════════════════════════════════════════════
# 6.  FULL PIPELINE (single entry-point)
# ════════════════════════════════════════════════════════════════

def run_preprocessing(
    db_path: str = "inventory.db",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Execute the complete preprocessing pipeline.

    Returns
    -------
    dict with keys:
        X_train, X_test, y_train, y_test, processed_df, conn
    """
    conn        = get_connection(db_path)
    raw_df      = load_vendor_invoice(conn)
    preview_data(raw_df)
    clean_df    = clean(raw_df)
    eng_df      = engineer_features(clean_df)
    X_train, X_test, y_train, y_test = split_data(
        eng_df, test_size, random_state
    )

    logger.info("Preprocessing pipeline complete.")
    return {
        "X_train"      : X_train,
        "X_test"       : X_test,
        "y_train"      : y_train,
        "y_test"       : y_test,
        "processed_df" : eng_df,
        "conn"         : conn,
    }


# ─── entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    data = run_preprocessing()
    print("\nX_train:\n", data["X_train"].head())
    print("\ny_train (Freight):\n", data["y_train"].head())
    print("\nProcessed df shape:", data["processed_df"].shape)
    data["conn"].close()