"""
invoice_data_preprocessing.py
==============================
Data ingestion, cleaning, feature engineering, labelling, and
train-test splitting for the Invoice Risk Flagging project.

Source tables : vendor_invoice  +  purchases  (SQLite → inventory.db)
Target column : flag_invoice  (0 = normal, 1 = risky)
Feature cols  : invoice_quantity, invoice_dollars, Freight,
                days_po_to_invoice, days_to_pay, total_brands,
                total_item_quantity, total_item_dollars,
                avg_receiving_delay

Author : Abhishek Thakur 
Project: Invoice Risk Flagging
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler

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
    db_path : str — path to the .db file

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

def load_raw_tables(
    conn: sqlite3.Connection,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load vendor_invoice and purchases tables.

    Returns
    -------
    invoice_df, purchases_df
    """
    invoice_df   = pd.read_sql("SELECT * FROM vendor_invoice", conn)
    purchases_df = pd.read_sql("SELECT * FROM purchases",      conn)

    logger.info("vendor_invoice  →  %d rows × %d cols", *invoice_df.shape)
    logger.info("purchases       →  %d rows × %d cols", *purchases_df.shape)

    return invoice_df, purchases_df


def preview_table(df: pd.DataFrame, name: str) -> None:
    """Log a quick snapshot of a DataFrame."""
    logger.info("\n[%s] Head:\n%s", name, df.head())
    logger.info("\n[%s] Dtypes:\n%s", name, df.dtypes)


# ════════════════════════════════════════════════════════════════
# 3.  SQL-EQUIVALENT JOIN & DATE CALCULATIONS
# ════════════════════════════════════════════════════════════════

def _cast_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert listed columns to datetime (errors → NaT)."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def build_joined_dataset(
    invoice_df:   pd.DataFrame,
    purchases_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replicate the notebook's CTE LEFT JOIN logic in Python.

    CTE (purchase_agg) computes per-PONumber:
        total_brands         — COUNT(DISTINCT Brand)
        total_item_quantity  — SUM(Quantity)
        total_item_dollars   — SUM(Dollars)
        avg_receiving_delay  — AVG(DATEDIFF(ReceivingDate, PODate))

    Then joins to vendor_invoice and adds:
        days_po_to_invoice   — InvoiceDate - PODate
        days_to_pay          — PayDate - InvoiceDate

    Returns
    -------
    Joined DataFrame ready for cleaning.
    """
    # ── date casts ──────────────────────────────────────────────
    invoice_df   = _cast_dates(invoice_df,   ["InvoiceDate", "PODate", "PayDate"])
    purchases_df = _cast_dates(purchases_df, ["ReceivingDate", "PODate"])

    # ── receiving delay ─────────────────────────────────────────
    purchases_df["receiving_delay_days"] = (
        (purchases_df["ReceivingDate"] - purchases_df["PODate"]).dt.days
    )

    # ── aggregate purchases by PO ────────────────────────────────
    purchase_agg = (
        purchases_df.groupby("PONumber")
        .agg(
            total_brands       =("Brand",                "nunique"),
            total_item_quantity=("Quantity",             "sum"),
            total_item_dollars =("Dollars",              "sum"),
            avg_receiving_delay=("receiving_delay_days", "mean"),
        )
        .reset_index()
    )
    logger.info("purchase_agg  →  %d unique PO numbers", len(purchase_agg))

    # ── join ────────────────────────────────────────────────────
    df = invoice_df.merge(purchase_agg, on="PONumber", how="left")

    # ── invoice-level date differences ──────────────────────────
    df["days_po_to_invoice"] = (df["InvoiceDate"] - df["PODate"]).dt.days
    df["days_to_pay"]        = (df["PayDate"]     - df["InvoiceDate"]).dt.days

    # ── rename for clarity ───────────────────────────────────────
    df = df.rename(columns={
        "Quantity": "invoice_quantity",
        "Dollars" : "invoice_dollars",
    })

    logger.info("Joined dataset  →  %d rows × %d cols", *df.shape)
    return df


# ════════════════════════════════════════════════════════════════
# 4.  CLEANING
# ════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────
# IMPORTANT — feature / label separation
# ─────────────────────────────────────────────────────────────────
# The risk label (flag_invoice) is derived from two rule-based
# signals:
#   Rule 1 → |invoice_dollars − total_item_dollars| > 5
#   Rule 2 → avg_receiving_delay > 10
#
# Those two columns CANNOT be model inputs — using them would let
# the model trivially reconstruct the labelling rules (label
# leakage), giving inflated metrics that mean nothing in production.
#
# Safe features (no direct algebraic link to the label):
#   invoice_quantity  — order size
#   Freight           — shipping cost
#   days_po_to_invoice — procurement lead time
#   days_to_pay        — payment behaviour
#   total_brands       — vendor diversity per PO
#   total_item_quantity — PO volume
# ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "invoice_quantity",
    "Freight",
    "days_po_to_invoice",
    "days_to_pay",
    "total_brands",
    "total_item_quantity",
]
TARGET_COL = "flag_invoice"


def report_missing(df: pd.DataFrame) -> pd.Series:
    """Log and return missing-value counts."""
    missing = df.isnull().sum()
    pct     = (missing / len(df) * 100).round(2)
    report  = pd.DataFrame({"count": missing, "pct": pct})
    has_missing = report[report["count"] > 0]
    if has_missing.empty:
        logger.info("No missing values found.")
    else:
        logger.warning("Missing values:\n%s", has_missing)
    return missing


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        logger.warning("Removed %d duplicate rows.", removed)
    else:
        logger.info("No duplicates found.")
    return df


def drop_nulls_in_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where any feature column is NaN."""
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS)
    logger.info(
        "dropna on features: %d → %d rows (removed %d)",
        before, len(df), before - len(df),
    )
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: dedup → missing report → dropna."""
    df = remove_duplicates(df)
    report_missing(df)
    df = drop_nulls_in_features(df)
    logger.info("Cleaning complete  →  %d rows remain.", len(df))
    return df


# ════════════════════════════════════════════════════════════════
# 5.  RISK LABEL CREATION
# ════════════════════════════════════════════════════════════════

# Tunable thresholds (used ONLY for labelling — NOT model features)
DOLLAR_MISMATCH_THRESHOLD = 5    # |invoice_dollars − total_item_dollars| > this → risky
RECEIVING_DELAY_THRESHOLD = 10   # avg_receiving_delay (days) > this → risky


def create_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a binary risk label to each invoice using vectorized
    pandas operations (fast, no row-wise apply).

    Rules
    -----
    Rule 1 — Dollar mismatch: |invoice_dollars − total_item_dollars| > 5
    Rule 2 — Slow receiving : avg_receiving_delay > 10 days

    NOTE: The columns used here (invoice_dollars, total_item_dollars,
    avg_receiving_delay) are intentionally EXCLUDED from FEATURE_COLS
    to prevent label leakage.

    Adds column: flag_invoice (0 = normal, 1 = risky)
    """
    df = df.copy()

    rule1 = (df["invoice_dollars"] - df["total_item_dollars"]).abs() > DOLLAR_MISMATCH_THRESHOLD
    rule2 = df["avg_receiving_delay"] > RECEIVING_DELAY_THRESHOLD

    df[TARGET_COL] = (rule1 | rule2).astype(int)

    dist  = df[TARGET_COL].value_counts()
    normal_count = dist.get(0, 0)
    risky_count  = dist.get(1, 0)
    logger.info(
        "Risk label distribution:\n  Normal (0): %d\n  Risky  (1): %d",
        normal_count, risky_count,
    )

    if normal_count > 0 and risky_count > 0:
        ratio = min(normal_count, risky_count) / max(normal_count, risky_count)
        if ratio < 0.2:
            logger.warning(
                "Class imbalance detected (ratio=%.2f). "
                "class_weight='balanced' is applied in the classifiers "
                "to compensate — see invoice_train.py.",
                ratio,
            )
    return df


# ════════════════════════════════════════════════════════════════
# 6.  TRAIN-TEST SPLIT + SCALING
# ════════════════════════════════════════════════════════════════

def split_and_scale(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[
    pd.DataFrame, pd.DataFrame,   # X_train, X_test
    pd.DataFrame, pd.DataFrame,   # X_train_scaled, X_test_scaled
    pd.Series,    pd.Series,      # y_train, y_test
    StandardScaler,
]:
    """
    Stratified train-test split + StandardScaler fit on train only.

    Returns
    -------
    X_train, X_test,
    X_train_scaled, X_test_scaled,
    y_train, y_test,
    scaler
    """
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,                   # preserve class ratio in both sets
    )

    scaler         = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURE_COLS
    )
    X_test_scaled  = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURE_COLS
    )

    logger.info(
        "Train-test split (%.0f/%.0f)  →  train: %d  |  test: %d",
        (1 - test_size) * 100, test_size * 100,
        len(X_train), len(X_test),
    )
    logger.info(
        "Train label dist:\n%s",
        y_train.value_counts().rename({0: "Normal", 1: "Risky"}),
    )
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ════════════════════════════════════════════════════════════════
# 7.  FULL PIPELINE
# ════════════════════════════════════════════════════════════════

def run_preprocessing(
    db_path: str = "inventory.db",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Execute the complete invoice preprocessing pipeline.

    Returns
    -------
    dict with keys:
        X_train, X_test,
        X_train_scaled, X_test_scaled,
        y_train, y_test,
        processed_df, scaler, conn
    """
    conn = get_connection(db_path)

    invoice_df, purchases_df = load_raw_tables(conn)
    preview_table(invoice_df,   "vendor_invoice")
    preview_table(purchases_df, "purchases")

    joined_df   = build_joined_dataset(invoice_df, purchases_df)
    clean_df    = clean(joined_df)
    labelled_df = create_risk_labels(clean_df)

    (
        X_train, X_test,
        X_train_scaled, X_test_scaled,
        y_train, y_test,
        scaler,
    ) = split_and_scale(labelled_df, test_size, random_state)

    logger.info("Preprocessing pipeline complete.")

    return {
        "X_train"        : X_train,
        "X_test"         : X_test,
        "X_train_scaled" : X_train_scaled,
        "X_test_scaled"  : X_test_scaled,
        "y_train"        : y_train,
        "y_test"         : y_test,
        "processed_df"   : labelled_df,
        "scaler"         : scaler,
        "conn"           : conn,
    }


# ─── entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    data = run_preprocessing()
    print("\nX_train shape    :", data["X_train"].shape)
    print("X_test  shape    :", data["X_test"].shape)
    print("Label distribution:\n",
          data["processed_df"]["flag_invoice"].value_counts())
    data["conn"].close()