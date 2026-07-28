"""Return Intelligence Platform — main application entry point."""
import streamlit as st
import pandas as pd
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Return Intelligence Platform",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

from components import GLOBAL_CSS
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    features   = pd.read_csv('data/features_scored.csv')
    sellers    = pd.read_csv('data/seller_risk_scores.csv')
    products   = pd.read_csv('data/products.csv')
    returns_df = pd.read_csv('data/returns.csv')
    sellers_info = pd.read_csv('data/sellers.csv')
    features['delay_bucket'] = pd.cut(
        features['delivery_delay_days'],
        bins=[-1, 0, 1, 2, 3, 5, 100],
        labels=['On time', '1 day', '2 days', '3 days', '4-5 days', '5+ days']
    )
    features['rating_bucket'] = pd.cut(
        features['rating'],
        bins=[0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.1],
        labels=['< 2.5', '2.5-3.0', '3.0-3.5', '3.5-4.0', '4.0-4.5', '4.5+']
    )
    return features, sellers, products, returns_df, sellers_info

@st.cache_resource
def load_model():
    model = joblib.load('models/xgb_model.pkl')
    with open('models/model_features.json') as f:
        cols = json.load(f)
    return model, cols

features, sellers, products, returns_df, sellers_info = load_data()
model, MODEL_FEATURES = load_model()

# ── SIDEBAR NAVIGATION ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 16px">
      <div style="display:flex;align-items:center;gap:8px">
        <div style="width:8px;height:8px;border-radius:50%;background:#1a73e8"></div>
        <span style="font-size:14px;font-weight:600;color:#fff">Return Intelligence</span>
      </div>
      <div style="font-size:11px;color:rgba(255,255,255,0.4);margin-top:2px;padding-left:16px">
        Operations Platform</div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigation",
        ["Overview", "Order Risk Analysis", "Seller Intelligence",
         "Scenario Planner", "Batch Order Scorer",
         "Customer Segments", "Model Monitoring"],
        label_visibility="collapsed"
    )

    st.divider()

    # Sidebar stats
    return_rate = features['is_returned'].mean() * 100
    st.markdown(f"""
    <div style="font-size:11px;color:rgba(255,255,255,0.35);line-height:1.8">
      XGBoost · AUC 0.892<br>
      {len(features):,} orders · {return_rate:.1f}% return rate<br>
      Data: Jan–Dec 2024
    </div>""", unsafe_allow_html=True)

# ── PAGE ROUTING ──────────────────────────────────────────────────────────────
if page == "Overview":
    from pages_mod.overview import render
    render(features, sellers, returns_df, sellers_info)

elif page == "Order Risk Analysis":
    from pages_mod.order_risk import render
    render(features, model, MODEL_FEATURES)

elif page == "Seller Intelligence":
    from pages_mod.seller_intel import render
    render(features, sellers, sellers_info, returns_df)

elif page == "Scenario Planner":
    from pages_mod.scenario import render
    render(features, returns_df)

elif page == "Batch Order Scorer":
    from pages_mod.batch_scorer import render
    render(features, model, MODEL_FEATURES)

elif page == "Customer Segments":
    from pages_mod.customer_segments import render
    render(features, returns_df)

elif page == "Model Monitoring":
    from pages_mod.model_monitor import render
    render(features)
