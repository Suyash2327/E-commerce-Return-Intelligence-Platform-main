<div align="center">

<img src="https://img.shields.io/badge/Status-Live%20Demo%20Ready-brightgreen?style=for-the-badge" />
<img src="https://img.shields.io/badge/Domain-E--Commerce%20Operations-blue?style=for-the-badge" />
<img src="https://img.shields.io/badge/ML%20Model-XGBoost%20AUC%200.892-orange?style=for-the-badge" />

<br/><br/>

# 📦 E-Commerce Return Intelligence Platform

### *From raw transaction data → enterprise operations tool that saves crores*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-ED8B00?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Engine-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

</div>

---

## 🚨 The Business Problem

> In e-commerce, **every returned order costs money twice** — once to ship out, and again to process the return.

This platform is built on a dataset of **200,000 real orders (Jan–Dec 2024)**. The numbers tell a stark story:

| Metric | Value |
|---|---|
| 📦 Total Orders | 2,00,000 |
| 🔁 Return Rate | **25.7%** (51,359 returns) |
| 💸 Avg Cost per Return | **₹5,271** |
| 🔥 Total Return Cost | **₹27.07 Crore** |
| ⚠️ GMV at Risk (High-Risk Orders) | **₹202.53 Crore** |

This platform uses historical data + a trained XGBoost model to help operations teams **identify, analyse, and prevent returns before they happen.**

---

## 🧠 Machine Learning at the Core

The predictive engine is a trained **XGBoost Classifier** that scores every order's return probability (0–100%) before dispatch.

| Metric | Value | Context |
|---|---|---|
| Algorithm | XGBoost Classifier | Gradient boosting with early stopping |
| Training Data | 2,00,000 orders | Jan–Dec 2024 |
| **AUC-ROC** | **0.892** | vs 0.739 logistic regression baseline |
| Recall @ 0.30 | **95%** | Catches nearly all expensive returns |
| Precision @ 0.30 | 46% | Acceptable trade-off for low-friction interventions |
| Features Used | 29 engineered features | Customer, seller, product, order signals |

**Key insight:** The model was deliberately tuned for high Recall (not Precision) because in the returns domain, missing a high-risk return is far more expensive than flagging a false positive.

---

## ✨ Platform Pages & Features

### `1` 📊 Operations Overview
The command centre. Every KPI is **dynamically calculated from the real dataset** — nothing is hardcoded.
- Portfolio-level KPI cards (total orders, return rate, return cost, GMV at risk, flagged sellers)
- Top 5 highest-risk sellers requiring immediate review
- Return rate breakdown by product category and payment method
- Trend charts and key analytical findings

---

### `2` 🎯 Order Risk Analysis *(Real-time ML Inference)*
An operational tool for customer service and fraud teams to assess any order **in real-time**.
- Input order parameters (category, price, payment method, seller rating, delivery delay, etc.)
- Instantly get a **return probability score** from the XGBoost model
- Understand *why* the order is risky via top **Risk Drivers** (e.g., COD payment = +8% return rate)
- Get a clear **Recommended Action** (block, flag, or standard processing)

---

### `3` 🏢 Seller Intelligence
Identify which merchants are destroying platform margins.
- Full seller risk table with return rate, risk score, and tier classification
- Filter by **High / Medium / Low Risk** tier
- Financial impact column — see exactly how much each seller costs in returns
- Sort, filter, and **export to CSV** for seller management teams
- Trend analysis: Platform Fulfilled vs Seller Fulfilled return rates over time

---

### `4` 📈 Scenario Planner *(ROI Simulator)*
The most powerful page — turns model insights into **a business case for action**.

Simulate 4 real interventions and get instant projections:
| Intervention | What It Does |
|---|---|
| 🔒 COD Enforcement | Force prepaid for high-risk COD customers |
| ⭐ Seller Quality Gate | Warn / suspend low-rated sellers |
| 🚚 Delivery SLA Policy | Penalise delayed deliveries |
| 📸 Image Policy | Reject listings with < 3 product images |

For each active intervention, the planner shows: **Returns Prevented**, **Gross Savings**, **Implementation Cost**, **Net ROI%**, and **Payback Period** in months.

---

### `5` 📂 Batch Order Scorer *(Bulk Inference)*
For operations teams that need to score hundreds of orders at once.
- Upload any CSV of orders (up to 10,000 rows)
- Download a pre-filled **CSV template** to get started instantly
- Get return probability, risk tier, top risk factor, and recommended action for **every row**
- Filter results by risk tier, set probability thresholds, and **download scored output**
- Visual summary: probability histogram, risk tier breakdown chart, top risk factors chart

---

### `6` 👤 Customer Behaviour Segmentation *(RFM for Returns)*
Goes beyond scoring orders — **scores customers** to detect return abuse patterns.

Segments all 41,006 unique customers into 4 behavioural profiles using aggregated order history:

| Segment | Rule | Platform Policy |
|---|---|---|
| 🔴 **Serial Returner** | Return rate > 50%, 3+ orders | Reduce return window to 7 days |
| 👻 **Ghost Buyer** | Returns nearly everything, new account | Force prepaid on all future orders, block COD |
| 🟡 **Occasional Returner** | Normal 1–2 returns, fit/size issues | Standard policy, send size guides |
| 🟢 **Loyal Low-Risk** | 4+ orders, return rate < 10% | Fast-track returns, no questions asked |

Features: Segment distribution donut, return-rate bar chart, interactive customer scatter plot, filterable drill-down table, full CSV export of 41,000+ customer profiles.

---

### `7` 🤖 Model Monitoring
For technical stakeholders and data science reviewers.
- Probability score distribution across all 200k orders
- Precision / Recall tradeoff curve across thresholds
- XGBoost Feature Importance (Gain) — shows what the model actually learned
- Dataset metadata and model health indicators

---

## 🗂️ Project Architecture

```
E-commerce-Return-Intelligence-Platform/
│
├── app.py                        # Streamlit orchestrator — routing & data loading
├── components.py                 # Design system: metric cards, charts, CSS tokens
├── requirements.txt              # All Python dependencies
│
├── .streamlit/
│   └── config.toml               # Light theme configuration
│
├── pages_mod/                    # One file per page (modular architecture)
│   ├── overview.py               # Operations Overview
│   ├── order_risk.py             # Real-time ML inference
│   ├── seller_intel.py           # Seller risk workspace
│   ├── scenario.py               # ROI scenario planner
│   ├── batch_scorer.py           # Bulk CSV scoring
│   ├── customer_segments.py      # Customer RFM segmentation
│   └── model_monitor.py          # ML performance monitoring
│
├── data/
│   ├── features_scored.csv       # 200k orders with ML predictions (57 features)
│   ├── returns.csv               # Return cost basis (avg ₹5,271)
│   ├── sellers.csv               # Seller metadata
│   ├── seller_risk_scores.csv    # Pre-aggregated seller risk scores
│   └── products.csv              # Product catalogue
│
└── models/
    ├── xgb_model.pkl             # Serialised XGBoost model
    └── model_features.json       # Feature schema (29 features)
```

---

## 🚀 Run It Locally

```bash
# 1. Clone
git clone https://github.com/HarshitGupta00/E-commerce-Return-Intelligence-Platform.git
cd E-commerce-Return-Intelligence-Platform

# 2. Install
pip install -r requirements.txt

# 3. Launch
streamlit run app.py
```
> App opens automatically at **`http://localhost:8501`**

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit 1.35 + Custom CSS Design System |
| Machine Learning | XGBoost, Scikit-Learn |
| Data Visualisation | Plotly Graph Objects & Express |
| Data Processing | Pandas, NumPy |
| Model Serialisation | Joblib |
| Language | Python 3.10+ |

---

## 💡 Why This Project Stands Out

> Most data science portfolios stop at "my model has 90% accuracy." This platform goes further.

- ✅ **End-to-end product thinking** — from raw CSV to a 7-page internal operations tool
- ✅ **Real business ROI** — every metric is calculated from actual data, not mocked
- ✅ **Customer-level abuse detection** — a feature real e-commerce platforms actually build
- ✅ **Modular, production-style code** — clean separation of concerns across pages
- ✅ **Non-technical accessible** — a business manager can use this without touching the model

---

<div align="center">
  <sub>Built by <strong>Harshit Gupta</strong> · Bridging Data Science & Business Operations</sub>
</div>
