# Vendor Invoice Project

End-to-end data science project covering SQL analytics, freight cost
prediction (regression), and invoice risk flagging (classification)
on an inventory/purchasing dataset stored in SQLite.

---

## Project structure

```
vendor_invoice_project/
│
├── notebooks/
│   ├── predicting_freight_cost.ipynb   # EDA + regression notebook
│   ├── invoice_flagging.ipynb          # EDA + classification notebook
│   └── inventory.db                   # SQLite database (all source tables)
│
├── freight_cost_prediction/
│   ├── data_preprocessing.py           # Load, clean, feature-engineer, split
│   ├── train.py                        # Train & select best regression model
│   └── model_evaluation.py            # Residual diagnostics, R², plots
│
├── invoice_flagging/
│   ├── data_processing.py             # Load, join, label, split
│   ├── train.py                       # Train & tune classification models
│   └── model_evaluation.py           # ROC, PR curve, confusion matrix, t-tests
│
├── vendor_sql.sql                     # SQL analysis queries
├── requirements.txt
└── README.md
```

---

## Projects

### 1. Predicting Freight Cost (Regression)

**Goal:** Predict the freight cost for a vendor invoice given the order
quantity and invoice dollar value.

**Source table:** `vendor_invoice`

**Features:** `Quantity`, `Dollars`

**Target:** `Freight`

**Models compared:** Linear Regression, Decision Tree, Random Forest

**Best result:** Random Forest — R² ≈ 0.97

**Key insight:** Bulk orders (high quantity) have significantly lower
freight cost per unit, confirming economies of scale in shipping.

---

### 2. Invoice Risk Flagging (Classification)

**Goal:** Flag invoices as risky (1) or normal (0) based on purchasing
behaviour patterns.

**Source tables:** `vendor_invoice` joined with `purchases`

**Labelling rules (rule-based, not model inputs):**
- Dollar mismatch: `|invoice_dollars − total_item_dollars| > $5`
- Receiving delay: `avg_receiving_delay > 10 days`

**Features (leakage-free):**
`invoice_quantity`, `Freight`, `days_po_to_invoice`, `days_to_pay`,
`total_brands`, `total_item_quantity`

> **Note on methodology:** Labels are derived from business rules
> applied to the data. The labelling columns are intentionally
> excluded from model features to prevent leakage. In a production
> setting, labels would come from audited invoice records or
> confirmed fraud/dispute cases.

**Models compared:** Logistic Regression, Decision Tree, Random Forest
(all with `class_weight='balanced'`)

**Tuning:** GridSearchCV on Decision Tree (cv=5, scoring=f1)

**Evaluation:** Confusion matrix, ROC-AUC, Precision-Recall curve,
feature importance, Welch t-tests (flagged vs normal)

---

### 3. SQL Analysis (`vendor_sql.sql`)

Analytical queries covering:
- Top vendors by purchase value
- Inventory movement (beginning vs ending)
- Most purchased products
- Freight cost efficiency by vendor
- Profit margin (purchase price vs retail price)
- ABC / Pareto analysis (80/20 rule)
- Vendor dependency analysis
- Inventory turnover indicator
- Sales vs inventory risk (estimated sales formula)

---

## Setup

### Requirements

- Python 3.10+
- See `requirements.txt`

### Install dependencies

```bash
pip install -r requirements.txt
```

### Database

Place `inventory.db` in the `notebooks/` directory (or update the
`db_path` argument in each script).

The database contains these tables:
`vendor_invoice`, `purchases`, `begin_inventory`, `end_inventory`,
`purchase_prices`

---

## Running the pipelines

### Freight cost prediction

```bash
cd freight_cost_prediction/

# Preprocess only
python data_preprocessing.py

# Train all models and save best
python train.py

# Evaluate saved model
python model_evaluation.py
```

### Invoice risk flagging

```bash
cd invoice_flagging/

# Preprocess only
python data_processing.py

# Train, tune, and save best model
python train.py

# Evaluate saved model
python model_evaluation.py
```

Saved models are written to `models/` in each sub-folder.
Evaluation plots are written to `evaluation_plots/`.

---

## Results summary

| Project | Model | Key Metric |
|---|---|---|
| Freight cost | Random Forest | R² = 0.97 |
| Invoice flagging | Tuned Decision Tree | F1 (risky) — see evaluation output |

---

## Author

**Abhishek Thakur**