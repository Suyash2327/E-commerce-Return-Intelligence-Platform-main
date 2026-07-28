"""Model Monitoring — technical ML evaluation and feature importance."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from components import (page_header, metric_card, section_label, chart_container,
                        callout_box, styled_plotly, COLORS, FEATURE_LABELS)


def render(features):
    page_header("Model Monitoring",
                "XGBoost evaluation, threshold analysis, and feature importance")

    # ── Model metadata ────────────────────────────────────────────────────
    section_label("Model Information")
    with st.expander("Model version and training details", expanded=False):
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown("**Algorithm:** XGBoost")
        with m2:
            st.markdown(f"**Training set:** {len(features):,} orders")
        with m3:
            st.markdown("**Data period:** Jan–Dec 2024")
        with m4:
            st.markdown("**Decision threshold:** 0.30")
        st.markdown("**Target variable:** `is_returned` (binary) · "
                    "**Evaluation:** Stratified holdout split · "
                    "**Positive class:** Returned orders")

    # ── Performance KPIs ──────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("AUC-ROC", "0.892",
                     "vs 0.739 baseline", delta_type='down',
                     border_color=COLORS['primary'])
    with k2:
        metric_card("Recall @ 0.30", "95%",
                     "Catches 95% of returns", delta_type='down',
                     border_color=COLORS['risk_low'])
    with k3:
        metric_card("Precision @ 0.30", "46%",
                     "Trade-off for high recall",
                     border_color=COLORS['risk_med'])
    with k4:
        metric_card("Average Precision", "0.743",
                     "vs 0.519 baseline", delta_type='down',
                     border_color=COLORS['primary'])

    st.markdown("")

    # ── Distribution + Threshold charts ───────────────────────────────────
    section_label("Probability Distribution & Threshold Analysis")
    c1, c2 = st.columns(2)

    with c1:
        chart_container("Return probability distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=features[features['is_returned'] == 0]['return_probability'],
            name='Not returned', nbinsx=50,
            marker_color='#188038', opacity=0.6))
        fig.add_trace(go.Histogram(
            x=features[features['is_returned'] == 1]['return_probability'],
            name='Returned', nbinsx=50,
            marker_color='#d93025', opacity=0.6))
        fig.add_vline(x=0.3, line_dash='dot', line_color='#202124',
                      annotation_text='Threshold 0.30',
                      annotation_font_size=10, annotation_font_color='#5f6368')
        styled_plotly(fig, height=300, barmode='overlay',
                      xaxis_title='Return probability', yaxis_title='Count',
                      showlegend=True)

    with c2:
        chart_container("Precision · Recall · F1 vs threshold")
        thresholds = np.arange(0.1, 0.9, 0.05)
        yt = features['is_returned'].values
        yp = features['return_probability'].values
        precs, recs, f1s = [], [], []
        for t in thresholds:
            pd_ = (yp >= t).astype(int)
            tp = ((pd_ == 1) & (yt == 1)).sum()
            fp = ((pd_ == 1) & (yt == 0)).sum()
            fn = ((pd_ == 0) & (yt == 1)).sum()
            pr = tp / (tp + fp) if (tp + fp) > 0 else 0
            rc = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0
            precs.append(round(pr, 3))
            recs.append(round(rc, 3))
            f1s.append(round(f1, 3))

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=thresholds, y=precs, name='Precision',
                                  line=dict(color='#1a73e8', width=2)))
        fig2.add_trace(go.Scatter(x=thresholds, y=recs, name='Recall',
                                  line=dict(color='#d93025', width=2)))
        fig2.add_trace(go.Scatter(x=thresholds, y=f1s, name='F1 Score',
                                  line=dict(color='#e8710a', width=1.5, dash='dot')))
        fig2.add_vline(x=0.3, line_dash='dot', line_color='#202124',
                       annotation_text='Selected', annotation_font_size=10)
        styled_plotly(fig2, height=300, xaxis_title='Decision threshold',
                      yaxis_title='Score', showlegend=True,
                      yaxis=dict(range=[0, 1], gridcolor='#f1f3f4'))

    # ── Feature importance ────────────────────────────────────────────────
    section_label("Feature Importance (XGBoost Gain)")
    st.caption("Importance values represent relative model feature weights, not SHAP values. "
               "Higher values indicate features the model relies on more for predictions.")

    importance_data = pd.DataFrame({
        'Feature': ['cust_return_rate', 'prod_return_rate', 'is_cod', 'discount_pct',
                     'delivery_delay_days', 'cust_cod_ratio', 'category_enc',
                     'prod_avg_discount_given', 'seller_return_rate', 'rating',
                     'account_age_days', 'final_price', 'payment_method_enc', 'unit_price'],
        'Importance': [2.18, 1.73, 0.11, 0.10, 0.09, 0.07, 0.06, 0.05, 0.04, 0.03,
                       0.02, 0.02, 0.02, 0.01]
    })
    importance_data['Label'] = importance_data['Feature'].map(FEATURE_LABELS).fillna(importance_data['Feature'])
    importance_data = importance_data.sort_values('Importance')

    fig3 = go.Figure(go.Bar(
        x=importance_data['Importance'], y=importance_data['Label'],
        orientation='h', marker_color='#1a73e8',
        text=[f'{v:.2f}' for v in importance_data['Importance']],
        textposition='outside', textfont=dict(size=10)))
    styled_plotly(fig3, height=380, xaxis_title='Relative importance', yaxis_title='',
                  margin=dict(l=0, r=50, t=10, b=0))

    # ── Threshold rationale ───────────────────────────────────────────────
    section_label("Threshold Decision Rationale")
    t1, t2, t3 = st.columns(3)
    with t1:
        callout_box("<strong>Business logic:</strong> A missed return (false negative) costs "
                    "an average of ₹5,271 in reverse logistics. A false positive adds minor "
                    "customer friction. We optimise for high recall to minimise missed returns.")
    with t2:
        callout_box("<strong>At threshold 0.30:</strong> Recall = 95% means we catch 95 out "
                    "of every 100 actual returns before they happen. Precision = 46% means "
                    "roughly half of flagged orders actually return — acceptable for a "
                    "pre-emptive system.")
    with t3:
        callout_box("<strong>Tuning guidance:</strong> If the intervention is high-friction "
                    "(e.g. blocking COD entirely), raise to 0.50 for higher precision. For "
                    "low-friction flags (notifications), keep at 0.30 for maximum coverage.")
