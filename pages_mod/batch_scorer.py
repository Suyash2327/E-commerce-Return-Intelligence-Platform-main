"""
pages_mod/batch_scorer.py
Batch Order Scorer — upload a CSV of orders, get return probability scores back.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

from components import (
    page_header, metric_card, section_label, callout_box,
    styled_plotly, COLORS, CHART_LAYOUT, format_inr
)


# ── Risk helpers ──────────────────────────────────────────────────────────────
def get_risk_tier(prob):
    if prob >= 0.60:   return 'High Risk'
    elif prob >= 0.30: return 'Medium Risk'
    return 'Low Risk'


def get_top_risk_factor(row):
    factors = []
    if row.get('is_cod', 0) == 1:            factors.append('COD payment')
    if row.get('rating', 5) < 3.5:           factors.append('Low seller rating')
    if row.get('delivery_delay_days', 0) >= 3: factors.append('Delivery delay')
    if row.get('discount_pct', 0) >= 40:     factors.append('High discount')
    if row.get('is_new_customer', 0) == 1:   factors.append('New customer')
    if row.get('is_low_image_count', 0) == 1: factors.append('Low image count')
    if row.get('is_high_price', 0) == 1:     factors.append('High price item')
    return factors[0] if factors else 'Category average'


# ── Scoring logic ─────────────────────────────────────────────────────────────
def score_batch(df_raw, model, MODEL_FEATURES, ref_features):
    """
    Score a user-uploaded CSV.
    Required columns: category, payment_method, unit_price, discount_pct,
                      delivery_delay_days, seller_rating, image_count,
                      account_age_days, fulfilment_type
    """
    df = df_raw.copy()
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    REQUIRED = ['category', 'payment_method', 'unit_price', 'discount_pct',
                'delivery_delay_days', 'seller_rating', 'image_count',
                'account_age_days', 'fulfilment_type']
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return None, missing

    # Reference averages
    avg_cust_return  = ref_features['cust_return_rate'].mean()
    cat_return_rates = ref_features.groupby('category')['prod_return_rate'].mean().to_dict()
    cat_map = {c: i for i, c in enumerate(sorted(ref_features['category'].unique()))}
    pay_map = {c: i for i, c in enumerate(sorted(ref_features['payment_method'].unique()))}
    seg_map = {c: i for i, c in enumerate(sorted(ref_features['customer_segment'].unique()))}
    ful_map = {c: i for i, c in enumerate(sorted(ref_features['fulfilment_type'].unique()))}

    results = []
    for _, row in df.iterrows():
        cat      = str(row.get('category', 'Electronics'))
        payment  = str(row.get('payment_method', 'COD'))
        price    = float(row.get('unit_price', 5000))
        discount = float(row.get('discount_pct', 10))
        delay    = float(row.get('delivery_delay_days', 0))
        rating   = float(row.get('seller_rating', 4.0))
        images   = int(row.get('image_count', 5))
        acc_age  = int(row.get('account_age_days', 180))
        fulfil   = str(row.get('fulfilment_type', 'Seller Fulfilled'))
        avg_pr   = cat_return_rates.get(cat, 0.22)

        inp = {
            'unit_price': price,
            'discount_pct': discount,
            'quantity': 1,
            'final_price': price * (1 - discount / 100),
            'delivery_delay_days': delay,
            'is_cod': int(payment == 'COD'),
            # The model was trained with this feature name — do not rename
            'is_flipkart_fulfilled': int(fulfil == 'Platform Fulfilled'),
            'avg_rating': 3.9,
            'image_count': images,
            'price_bucket': min(5, max(1, int(price / 30000 * 4) + 1)),
            'account_age_days': acc_age,
            'is_new_customer': int(acc_age < 30),
            'cust_total_orders': max(1, acc_age // 30),
            'cust_return_rate': avg_cust_return,
            'cust_cod_ratio': 1.0 if payment == 'COD' else 0.2,
            'rating': rating,
            'seller_return_rate': 0.25,
            'seller_avg_discount': discount,
            'prod_return_rate': avg_pr,
            'prod_avg_discount_given': discount,
            'is_high_discount': int(discount >= 40),
            'is_delayed': int(delay > 0),
            'is_low_rated_seller': int(rating < 3.5),
            'is_low_image_count': int(images <= 2),
            'is_high_price': int(price > 20000),
            'category_enc': cat_map.get(cat, 0),
            'customer_segment_enc': seg_map.get('Regular', 0),
            'payment_method_enc': pay_map.get(payment, 0),
            'fulfilment_enc': ful_map.get(fulfil, 0),
        }
        results.append(inp)

    X = pd.DataFrame(results)[MODEL_FEATURES].fillna(0)
    probs = model.predict_proba(X)[:, 1]

    df['return_probability'] = np.round(probs * 100, 1)
    df['risk_tier']          = [get_risk_tier(p) for p in probs]
    df['top_risk_factor']    = [get_top_risk_factor(r) for r in results]
    df['recommended_action'] = df['risk_tier'].map({
        'High Risk':   'Require prepaid · Quality check before dispatch',
        'Medium Risk': 'Flag for post-delivery follow-up',
        'Low Risk':    'Standard processing',
    })

    return df.sort_values('return_probability', ascending=False), None


# ── CSV template ──────────────────────────────────────────────────────────────
TEMPLATE_DATA = pd.DataFrame([
    {'category': 'Electronics', 'payment_method': 'COD',
     'unit_price': 45000, 'discount_pct': 15, 'delivery_delay_days': 2,
     'seller_rating': 3.2, 'image_count': 3, 'account_age_days': 20,
     'fulfilment_type': 'Seller Fulfilled'},
    {'category': 'Fashion', 'payment_method': 'UPI',
     'unit_price': 1200, 'discount_pct': 50, 'delivery_delay_days': 0,
     'seller_rating': 4.5, 'image_count': 7, 'account_age_days': 365,
     'fulfilment_type': 'Platform Fulfilled'},
    {'category': 'Books', 'payment_method': 'Credit Card',
     'unit_price': 450, 'discount_pct': 5, 'delivery_delay_days': 1,
     'seller_rating': 4.8, 'image_count': 4, 'account_age_days': 730,
     'fulfilment_type': 'Platform Fulfilled'},
])


# ── Main render ───────────────────────────────────────────────────────────────
def render(features, model, MODEL_FEATURES):
    page_header(
        "Batch Order Scorer",
        "Upload a CSV of orders — get return probability, risk tier, and recommended action for every row instantly"
    )

    # ── Instructions + Template download ─────────────────────────────────────
    col_info, col_dl = st.columns([2, 1])
    with col_info:
        callout_box(
            "<strong>How to use:</strong><br>"
            "1. Download the CSV template &nbsp;→&nbsp; Fill in your order data (one row per order)<br>"
            "2. Upload the filled CSV &nbsp;→&nbsp; The platform scores every order in seconds<br>"
            "3. Filter results and download the scored CSV for your team",
            style='info'
        )
    with col_dl:
        template_csv = TEMPLATE_DATA.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇  Download CSV template",
            data=template_csv,
            file_name='order_batch_template.csv',
            mime='text/csv',
            use_container_width=True
        )

    st.divider()

    # ── File uploader ─────────────────────────────────────────────────────────
    section_label("Upload Order Batch")
    uploaded = st.file_uploader(
        'Upload CSV file',
        type=['csv'],
        help='Max 10,000 rows. Required columns: category, payment_method, '
             'unit_price, discount_pct, delivery_delay_days, seller_rating, '
             'image_count, account_age_days, fulfilment_type',
        label_visibility='collapsed'
    )

    if uploaded is None:
        section_label("Expected CSV Format")
        st.dataframe(TEMPLATE_DATA, use_container_width=True, hide_index=True)
        return

    # ── Parse & score ─────────────────────────────────────────────────────────
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    if len(df_raw) == 0:
        st.error("Uploaded file is empty.")
        return

    if len(df_raw) > 10000:
        st.warning("File has more than 10,000 rows. Only the first 10,000 will be scored.")
        df_raw = df_raw.head(10000)

    with st.spinner(f"Scoring {len(df_raw):,} orders..."):
        scored_df, missing_cols = score_batch(df_raw, model, MODEL_FEATURES, features)

    if missing_cols:
        st.error(f"Missing required columns: **{', '.join(missing_cols)}**")
        st.info("Download the template above to see the correct column names.")
        return

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    total       = len(scored_df)
    high_risk   = int((scored_df['risk_tier'] == 'High Risk').sum())
    medium_risk = int((scored_df['risk_tier'] == 'Medium Risk').sum())
    low_risk    = int((scored_df['risk_tier'] == 'Low Risk').sum())
    avg_prob    = scored_df['return_probability'].mean()
    avg_return_cost = features['return_cost'].mean() if 'return_cost' in features.columns else 5271
    est_savings = high_risk * avg_return_cost * 0.25

    section_label("Batch Results")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        metric_card("Orders Scored", f"{total:,}", "Batch processed")
    with k2:
        metric_card("High Risk", f"{high_risk:,}", "Require intervention",
                    delta_type='up', border_color=COLORS['risk_high'])
    with k3:
        metric_card("Medium Risk", f"{medium_risk:,}", "Monitor closely",
                    border_color=COLORS['risk_med'])
    with k4:
        metric_card("Low Risk", f"{low_risk:,}", "Standard processing",
                    delta_type='down', border_color=COLORS['risk_low'])
    with k5:
        metric_card("Est. Savings if Actioned", format_inr(est_savings),
                    "25% reduction on high-risk", delta_type='down',
                    border_color=COLORS['primary'])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(
            x=scored_df['return_probability'],
            nbinsx=30,
            marker_color=COLORS['primary'],
            opacity=0.75,
            name='Orders'
        ))
        fig1.add_vline(x=30, line_dash='dot', line_color=COLORS['risk_med'],
                       annotation_text='Medium (30%)', annotation_font_size=10)
        fig1.add_vline(x=60, line_dash='dot', line_color=COLORS['risk_high'],
                       annotation_text='High (60%)', annotation_font_size=10)
        styled_plotly(fig1, height=250,
                      xaxis_title='Return Probability (%)', yaxis_title='Order Count',
                      showlegend=False, title='Return Probability Distribution')

    with c2:
        tier_counts = scored_df['risk_tier'].value_counts().reset_index()
        tier_counts.columns = ['tier', 'count']
        color_map = {'High Risk': COLORS['risk_high'],
                     'Medium Risk': COLORS['risk_med'],
                     'Low Risk': COLORS['risk_low']}
        fig2 = go.Figure(go.Bar(
            x=tier_counts['tier'],
            y=tier_counts['count'],
            marker_color=[color_map.get(t, '#9CA3AF') for t in tier_counts['tier']],
            text=tier_counts['count'],
            textposition='outside',
        ))
        styled_plotly(fig2, height=250,
                      xaxis_title='', yaxis_title='Orders',
                      showlegend=False, title='Risk Tier Breakdown')

    # Top risk factors
    factor_counts = scored_df['top_risk_factor'].value_counts().reset_index()
    factor_counts.columns = ['factor', 'count']
    fig3 = go.Figure(go.Bar(
        x=factor_counts['count'],
        y=factor_counts['factor'],
        orientation='h',
        marker_color=COLORS['primary'],
        text=factor_counts['count'],
        textposition='outside',
    ))
    styled_plotly(fig3, height=220,
                  xaxis_title='Number of orders', yaxis_title='',
                  showlegend=False, title='Top Risk Factors Across Batch',
                  margin=dict(l=0, r=60, t=28, b=0))

    # ── Filtered table ────────────────────────────────────────────────────────
    section_label("Scored Order List")

    f1, f2, f3 = st.columns(3)
    with f1:
        tier_filter = st.selectbox('Filter by risk tier',
                                   ['All', 'High Risk', 'Medium Risk', 'Low Risk'])
    with f2:
        min_prob = st.slider('Min. probability (%)', 0, 100, 0)
    with f3:
        show_n = st.selectbox('Show top N rows', [25, 50, 100, 250, 'All'])

    display_df = scored_df.copy()
    if tier_filter != 'All':
        display_df = display_df[display_df['risk_tier'] == tier_filter]
    display_df = display_df[display_df['return_probability'] >= min_prob]
    if show_n != 'All':
        display_df = display_df.head(int(show_n))

    display_cols = ['category', 'payment_method', 'unit_price', 'discount_pct',
                    'delivery_delay_days', 'seller_rating', 'return_probability',
                    'risk_tier', 'top_risk_factor', 'recommended_action']
    display_cols = [c for c in display_cols if c in display_df.columns]
    st.dataframe(display_df[display_cols], use_container_width=True, hide_index=True)

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    csv_out = scored_df[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇  Download scored results as CSV",
        data=csv_out,
        file_name='scored_orders.csv',
        mime='text/csv',
    )

    if high_risk > 0:
        callout_box(
            f"<strong>Action required:</strong> {high_risk:,} orders flagged as High Risk "
            f"(return probability ≥ 60%). Recommended: require prepaid payment or add a "
            f"pre-dispatch quality check. Estimated savings if actioned: "
            f"<strong>{format_inr(est_savings)}</strong>.",
            style='critical'
        )
