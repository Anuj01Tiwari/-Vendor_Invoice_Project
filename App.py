"""
app.py
======
Streamlit web app for the Vendor Invoice Project.

Pages
-----
  1. Invoice Risk Checker    — predict RISKY / NORMAL for an invoice
  2. Freight Cost Predictor  — predict freight cost from order details
  3. SQL Insights            — charts from vendor / inventory analysis

Run locally:
    streamlit run app.py

Author : Abhishek Thakur
Project: Vendor Invoice Project
"""

import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ─────────────────────────── page config ────────────────────────
st.set_page_config(
    page_title="Vendor Invoice Project",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── paths ──────────────────────────────
BASE_DIR        = Path(__file__).parent
MODELS_DIR      = BASE_DIR / "notebooks"
NOTEBOOKS_DIR   = BASE_DIR / "notebooks"

INVOICE_MODEL_PATH = MODELS_DIR / "invoice_flagging_model.pkl"
FREIGHT_MODEL_PATH = MODELS_DIR / "predicting_freight_cost_model.pkl"
DB_PATH            = NOTEBOOKS_DIR / "inventory.db"

# ─────────────────── feature columns ────────────────────────────
INVOICE_FEATURES = [
    "invoice_quantity",
    "Freight",
    "days_po_to_invoice",
    "days_to_pay",
    "total_brands",
    "total_item_quantity",
]

FREIGHT_FEATURES = ["Quantity", "Dollars"]

# ─────────────────── custom CSS ─────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #f8f9fa; }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
    }
    .badge-risky {
        background: #fff0f0;
        color: #c0392b;
        border: 1.5px solid #e74c3c;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        font-size: 1.3rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .badge-normal {
        background: #f0fff4;
        color: #1a7a3c;
        border: 1.5px solid #27ae60;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        font-size: 1.3rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .badge-freight {
        background: #eff6ff;
        color: #1a56db;
        border: 1.5px solid #3b82f6;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        font-size: 1.3rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.5rem;
    }
    hr { border: none; border-top: 1px solid #e9ecef; margin: 1.5rem 0; }
    .section-tag {
        background: #eef2ff;
        color: #4338ca;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    """Load and cache a sklearn model from disk."""
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def get_db_connection():
    """Open and cache a SQLite connection."""
    if not DB_PATH.exists():
        return None
    return sqlite3.connect(DB_PATH, check_same_thread=False)


@st.cache_data(show_spinner=False)
def run_query(sql: str) -> pd.DataFrame:
    """Execute a SQL query and return a DataFrame."""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    return pd.read_sql(sql, conn)


def model_missing_warning(name: str):
    st.warning(
        f"**{name} model not found.**  \n"
        f"Run `python train.py` inside the relevant sub-folder first, "
        f"then restart the app.",
        icon="⚠️",
    )


# ════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📦 Vendor Invoice")
    st.caption("Abhishek Thakur · Data Science Portfolio")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "🧾  Invoice Risk Checker",
            "🚚  Freight Cost Predictor",
            "📊  SQL Insights",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#888'>Models: sklearn Pipelines  \n"
        "Data: SQLite (inventory.db)  \n"
        "Stack: Python · Streamlit · scikit-learn</small>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════
# PAGE 1 — INVOICE RISK CHECKER
# ════════════════════════════════════════════════════════════════

if page == "🧾  Invoice Risk Checker":

    st.markdown('<span class="section-tag">Classification</span>', unsafe_allow_html=True)
    st.title("Invoice Risk Checker")
    st.markdown(
        "Enter invoice details below. The model will predict whether the "
        "invoice is **NORMAL** or **RISKY** based on purchasing behaviour patterns."
    )
    st.markdown("---")

    model = load_model(INVOICE_MODEL_PATH)

    if model is None:
        model_missing_warning("Invoice Flagging")
    else:
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.subheader("Invoice details")
            invoice_quantity   = st.number_input("Invoice quantity (units)",   min_value=0,   value=50,      step=1)
            invoice_dollars    = st.number_input("Invoice amount ($)",         min_value=0.0, value=1500.0,  step=100.0, format="%.2f")
            freight            = st.number_input("Freight cost ($)",           min_value=0.0, value=120.0,   step=10.0,  format="%.2f")
            days_po_to_invoice = st.number_input("Days from PO to invoice",    min_value=0,   value=5,       step=1)
            days_to_pay        = st.number_input("Days to pay invoice",        min_value=0,   value=30,      step=1)

        with col2:
            st.subheader("PO / vendor details")
            total_brands        = st.number_input("Number of brands on PO",    min_value=1,   value=3,       step=1)
            total_item_quantity = st.number_input("Total PO item quantity",    min_value=0,   value=200,     step=10)
            total_item_dollars  = st.number_input("Total PO item value ($)",   min_value=0.0, value=1400.0,  step=100.0, format="%.2f")
            avg_receiving_delay = st.number_input("Avg receiving delay (days)", min_value=0.0, value=6.0,    step=1.0,   format="%.1f")

            st.markdown("&nbsp;")
            predict_btn = st.button("Predict risk", type="primary", use_container_width=True)

        st.markdown("---")

        if predict_btn:
            input_vector = [[
                invoice_quantity,
                invoice_dollars,
                freight,
                days_po_to_invoice,
                days_to_pay,
                total_brands,
                total_item_quantity,
                total_item_dollars,
                avg_receiving_delay,
            ]]

            prediction = int(model.predict(input_vector)[0])
            proba      = None
            if hasattr(model, "predict_proba"):
                proba = float(model.predict_proba(input_vector)[0][1])

            res_col1, res_col2, res_col3 = st.columns(3)

            if prediction == 1:
                with res_col1:
                    st.markdown('<div class="badge-risky">RISKY INVOICE</div>', unsafe_allow_html=True)
                with res_col2:
                    if proba is not None:
                        st.metric("Risk probability", f"{proba:.1%}")
                with res_col3:
                    st.error(
                        "This invoice shows patterns associated with risky behaviour — "
                        "dollar mismatches or delayed receiving. Recommend manual review.",
                        icon="🚨",
                    )
            else:
                with res_col1:
                    st.markdown('<div class="badge-normal">NORMAL INVOICE</div>', unsafe_allow_html=True)
                with res_col2:
                    if proba is not None:
                        st.metric("Risk probability", f"{proba:.1%}")
                with res_col3:
                    st.success(
                        "No risk patterns detected. This invoice looks consistent "
                        "with normal purchasing behaviour.",
                        icon="✅",
                    )

        with st.expander("What does the model use? (feature reference)"):
            st.markdown("""
| Feature | Description |
|---|---|
| Invoice quantity | Number of units on this invoice |
| Freight cost | Shipping/logistics cost for this invoice |
| Days PO to invoice | Lead time: how long after PO was invoice raised |
| Days to pay | Payment speed: days between invoice and payment |
| Brands on PO | Vendor diversity — how many brands on the purchase order |
| Total PO quantity | Total volume of items on the linked purchase order |

**Note:** The labelling rules (dollar mismatch > $5, avg receiving delay > 10 days)
are intentionally excluded from model features to prevent label leakage.
            """)


# ════════════════════════════════════════════════════════════════
# PAGE 2 — FREIGHT COST PREDICTOR
# ════════════════════════════════════════════════════════════════

elif page == "🚚  Freight Cost Predictor":

    st.markdown('<span class="section-tag">Regression</span>', unsafe_allow_html=True)
    st.title("Freight Cost Predictor")
    st.markdown(
        "Enter your order details to predict how much the freight will cost. "
        "The model was trained on historical vendor invoice data."
    )
    st.markdown("---")

    model = load_model(FREIGHT_MODEL_PATH)

    if model is None:
        model_missing_warning("Freight Cost")
    else:
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.subheader("Order details")
            quantity = st.number_input(
                "Order quantity (units)",
                min_value=1, value=500, step=50,
                help="Total number of units in the order",
            )
            dollars = st.number_input(
                "Order value ($)",
                min_value=0.0, value=12000.0, step=500.0, format="%.2f",
                help="Total invoice dollar value before freight",
            )
            predict_btn = st.button("Predict freight cost", type="primary", use_container_width=True)

        with col2:
            st.subheader("What to expect")
            st.info(
                "Freight cost scales with order value and quantity. "
                "Larger bulk orders typically have a **lower freight cost per unit** "
                "(economies of scale).",
                icon="💡",
            )
            freight_per_unit_est = (dollars * 0.01) / max(quantity, 1)
            st.markdown(
                f"**Estimated freight/unit** (rough guide): "
                f"${freight_per_unit_est:.4f}"
            )

        st.markdown("---")

        if predict_btn:
            prediction = float(model.predict([[quantity, dollars]])[0])

            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.markdown(
                    f'<div class="badge-freight">Predicted freight: ${prediction:,.2f}</div>',
                    unsafe_allow_html=True,
                )
            with res_col2:
                freight_pct = (prediction / dollars * 100) if dollars > 0 else 0
                st.metric("Freight as % of order value", f"{freight_pct:.2f}%")
            with res_col3:
                freight_per_unit = prediction / quantity if quantity > 0 else 0
                st.metric("Freight per unit", f"${freight_per_unit:.4f}")

        st.markdown("---")
        st.subheader("Economies of scale — how quantity affects freight/unit")

        qty_range   = np.linspace(10, 2000, 200)
        dollars_mid = dollars if "dollars" in dir() else 12000
        preds       = model.predict(np.column_stack([qty_range, np.full_like(qty_range, dollars_mid)]))
        fpu         = preds / qty_range

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(qty_range, fpu, color="#3b82f6", linewidth=2)
        ax.fill_between(qty_range, fpu, alpha=0.1, color="#3b82f6")
        ax.set_xlabel("Order quantity (units)")
        ax.set_ylabel("Freight per unit ($)")
        ax.set_title(f"Freight per unit vs quantity  (order value = ${dollars_mid:,.0f})")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.3f"))
        sns.despine(ax=ax)
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ════════════════════════════════════════════════════════════════
# PAGE 3 — SQL INSIGHTS
# ════════════════════════════════════════════════════════════════

elif page == "📊  SQL Insights":

    st.markdown('<span class="section-tag">SQL Analysis</span>', unsafe_allow_html=True)
    st.title("SQL Insights")
    st.markdown(
        "Charts powered by the SQL queries in `vendor_sql.sql`, "
        "running live against `inventory.db`."
    )
    st.markdown("---")

    conn = get_db_connection()
    if conn is None:
        st.error(
            "**Database not found.**  \n"
            f"Expected: `{DB_PATH}`  \n"
            "Place `inventory.db` in the `notebooks/` folder and restart.",
            icon="🗄️",
        )
        st.stop()

    # ── Top vendors by purchase value ───────────────────────────
    st.subheader("Top 10 vendors by total purchase value")
    vendor_sql = """
        SELECT VendorName,
               SUM(Dollars)  AS total_purchase_value,
               SUM(Quantity) AS total_items_purchased
        FROM purchases
        GROUP BY VendorName
        ORDER BY total_purchase_value DESC
        LIMIT 10
    """
    vendor_df = run_query(vendor_sql)

    if not vendor_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.barh(
            vendor_df["VendorName"][::-1],
            vendor_df["total_purchase_value"][::-1] / 1e6,
            color="#4c72b0", edgecolor="white",
        )
        ax.set_xlabel("Total purchase value ($M)")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fM"))
        ax.set_title("Top 10 vendors by spend")
        sns.despine(ax=ax)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        with st.expander("View raw data"):
            vendor_df["total_purchase_value"] = vendor_df["total_purchase_value"].map("${:,.0f}".format)
            vendor_df["total_items_purchased"] = vendor_df["total_items_purchased"].map("{:,}".format)
            st.dataframe(vendor_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Freight cost efficiency ─────────────────────────────────
    st.subheader("Vendor freight cost efficiency (freight as % of invoice value)")
    freight_sql = """
        SELECT VendorName,
               SUM(Dollars)                                  AS total_invoice_value,
               SUM(Freight)                                  AS total_freight_cost,
               ROUND(SUM(Freight) / SUM(Dollars) * 100, 2)  AS freight_percentage
        FROM vendor_invoice
        GROUP BY VendorName
        HAVING SUM(Dollars) > 0
        ORDER BY freight_percentage DESC
        LIMIT 15
    """
    freight_df = run_query(freight_sql)

    if not freight_df.empty:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ["#c44e52" if v > freight_df["freight_percentage"].median() else "#55a868"
                      for v in freight_df["freight_percentage"]]
            ax.barh(
                freight_df["VendorName"][::-1],
                freight_df["freight_percentage"][::-1],
                color=colors[::-1], edgecolor="white",
            )
            ax.axvline(
                freight_df["freight_percentage"].median(),
                color="#888", linestyle="--", linewidth=1,
                label=f"Median: {freight_df['freight_percentage'].median():.1f}%",
            )
            ax.set_xlabel("Freight as % of invoice value")
            ax.legend(fontsize=9)
            ax.set_title("Freight cost efficiency by vendor")
            sns.despine(ax=ax)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_b:
            st.markdown("**Key insight**")
            high = freight_df[freight_df["freight_percentage"] > freight_df["freight_percentage"].median()]
            st.error(
                f"{len(high)} vendors have above-median freight costs. "
                f"These are candidates for shipping contract renegotiation.",
                icon="📍",
            )
            avg = freight_df["freight_percentage"].mean()
            st.metric("Average freight %", f"{avg:.2f}%")
            worst = freight_df.iloc[0]
            st.metric(
                "Highest freight %",
                f"{worst['freight_percentage']:.2f}%",
                delta=f"{worst['VendorName'][:20]}",
                delta_color="inverse",
            )

    st.markdown("---")

    # ── Most purchased products ─────────────────────────────────
    st.subheader("Top 10 most purchased products")
    products_sql = """
        SELECT Description,
               SUM(Quantity) AS total_qty,
               SUM(Dollars)  AS total_cost
        FROM purchases
        GROUP BY Description
        ORDER BY total_qty DESC
        LIMIT 10
    """
    prod_df = run_query(products_sql)

    if not prod_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].bar(
            range(len(prod_df)),
            prod_df["total_qty"],
            color="#4c72b0", edgecolor="white",
        )
        axes[0].set_xticks(range(len(prod_df)))
        axes[0].set_xticklabels(
            [d[:18] for d in prod_df["Description"]], rotation=40, ha="right", fontsize=8
        )
        axes[0].set_ylabel("Total units purchased")
        axes[0].set_title("By quantity")
        sns.despine(ax=axes[0])

        axes[1].bar(
            range(len(prod_df)),
            prod_df["total_cost"] / 1000,
            color="#dd8452", edgecolor="white",
        )
        axes[1].set_xticks(range(len(prod_df)))
        axes[1].set_xticklabels(
            [d[:18] for d in prod_df["Description"]], rotation=40, ha="right", fontsize=8
        )
        axes[1].set_ylabel("Total cost ($k)")
        axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fk"))
        axes[1].set_title("By purchase cost")
        sns.despine(ax=axes[1])

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")

    # ── Inventory change ────────────────────────────────────────
    st.subheader("Inventory change by store (beginning vs ending)")
    inv_sql = """
        SELECT b.Store,
               SUM(b.onHand)                      AS beginning_inventory,
               SUM(e.onHand)                      AS ending_inventory,
               SUM(e.onHand) - SUM(b.onHand)      AS inventory_change
        FROM begin_inventory b
        JOIN end_inventory e ON b.InventoryId = e.InventoryId
        GROUP BY b.Store
        ORDER BY inventory_change DESC
        LIMIT 15
    """
    inv_df = run_query(inv_sql)

    if not inv_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in inv_df["inventory_change"]]
        ax.bar(inv_df["Store"].astype(str), inv_df["inventory_change"], color=colors, edgecolor="white")
        ax.axhline(0, color="#888", linewidth=0.8)
        ax.set_xlabel("Store")
        ax.set_ylabel("Inventory change (units)")
        ax.set_title("Inventory gained (green) vs lost (red) by store")
        ax.tick_params(axis="x", rotation=45)
        sns.despine(ax=ax)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        gained = (inv_df["inventory_change"] > 0).sum()
        lost   = (inv_df["inventory_change"] < 0).sum()
        m1, m2, m3 = st.columns(3)
        m1.metric("Stores with stock gain",    gained)
        m2.metric("Stores with stock decline", lost)
        m3.metric(
            "Largest single gain",
            f"{inv_df['inventory_change'].max():+,.0f} units",
        )