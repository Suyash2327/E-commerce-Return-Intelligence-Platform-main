"""
pages_mod/batch_scorer.py
Batch Order Scorer — upload a CSV of orders, get return probability scores back
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import json
import io


# ── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('models/xgb_model.pkl')
    with open('models/model_features.json') as f:
        cols = json.load(f)
    return model, cols


@st.cache_data
def load_reference_data():
    features = pd.read_csv('data/features_scored.csv')
    return features


def get_risk_tier(prob):
    if prob >= 0.60: return 'High Risk'
    elif prob >= 0.30: return 'Medium Risk'
    return 'Low Risk'


def get_top_risk_factor(row):
    factors = []
    if row.get('is_cod', 0) == 1:
        factors.append('COD payment')
    if row.get('rating', 5) < 3.5:
        factors.append('Low seller rating')
    if row.get('delivery_delay_days', 0) >= 3:
        factors.append('Delivery delay')
    if row.get('discount_pct', 0) >= 40:
        factors.append('High discount')
    if row.get('is_new_customer', 0) == 1:
        factors.append('New customer')
    if row.get('is_low_image_count', 0) == 1:
        factors.append('Low image count')
    if row.get('is_high_price', 0) == 1:
        factors.append('High price item')
    return factors[0] if factors else 'Category average'


def score_batch(df_raw, model, MODEL_FEATURES, ref_features):
    """
    Score a user-uploaded CSV.
    Required columns: category, payment_method, unit_price, discount_pct,
                      delivery_delay_days, seller_rating, image_count,
                      account_age_days, fulfilment_type
    """
    df = df_raw.copy()

    # Column name normalisation
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # Check required columns
    REQUIRED = ['category', 'payment_method', 'unit_price',
                'discount_pct', 'delivery_delay_days',
                'seller_rating', 'image_count',
                'account_age_days', 'fulfilment_type']
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return None, missing

    # Reference averages for history features
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

        avg_pr = cat_return_rates.get(cat, 0.22)

        inp = {
            'unit_price': price,
            'discount_pct': discount,
            'quantity': 1,
            'final_price': price * (1 - discount / 100),
            'delivery_delay_days': delay,
            'is_cod': int(payment == 'COD'),
            'is_flipkart_fulfilled': int(fulfil == 'Flipkart Fulfilled'),
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


CHART_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=0, r=0, t=10, b=0),
    font=dict(family='system-ui', size=11, color='#374151'),
    xaxis=dict(gridcolor='#F3F4F6', linecolor='#E5E7EB'),
    yaxis=dict(gridcolor='#F3F4F6', linecolor='#E5E7EB'),
    showlegend=True,
    legend=dict(orientation='h', y=1.08, x=0, font=dict(size=11))
)


# ── Template download ─────────────────────────────────────────────────────────
TEMPLATE_DATA = pd.DataFrame([
    {
        'category': 'Electronics',
        'payment_method': 'COD',
        'unit_price': 45000,
        'discount_pct': 15,
        'delivery_delay_days': 2,
        'seller_rating': 3.2,
        'image_count': 3,
        'account_age_days': 20,
        'fulfilment_type': 'Seller Fulfilled',
    },
    {
        'category': 'Fashion',
        'payment_method': 'UPI',
        'unit_price': 1200,
        'discount_pct': 50,
        'delivery_delay_days': 0,
        'seller_rating': 4.5,
        'image_count': 7,
        'account_age_days': 365,
        'fulfilment_type': 'Flipkart Fulfilled',
    },
    {
        'category': 'Books',
        'payment_method': 'Credit Card',
        'unit_price': 450,
        'discount_pct': 5,
        'delivery_delay_days': 1,
        'seller_rating': 4.8,
        'image_count': 4,
        'account_age_days': 730,
        'fulfilment_type': 'Flipkart Fulfilled',
    },
])


# ── Main render function ──────────────────────────────────────────────────────
def render():
    model, MODEL_FEATURES = load_model()
    ref_features = load_reference_data()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown('<div class="page-title">Batch Order Scorer</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Upload a CSV of orders — get return probability, '
        'risk tier, and recommended action for every row instantly</div>',
        unsafe_allow_html=True)

    # ── Instructions + Template ───────────────────────────────────────────────
    col_info, col_dl = st.columns([2, 1])

    with col_info:
        st.markdown("""
        <div class="insight">
        <strong>How to use:</strong><br>
        1. Download the template CSV below<br>
        2. Fill in your order data — one row per order<br>
        3. Upload the filled CSV<br>
        4. The platform scores every order in seconds and lets you download the results
        </div>
        """, unsafe_allow_html=True)

    with col_dl:
        template_csv = TEMPLATE_DATA.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='⬇ Download CSV template',
            data=template_csv,
            file_name='order_batch_template.csv',
            mime='text/csv',
            use_container_width=True
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── File uploader ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Upload order batch</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        'Upload CSV file',
        type=['csv'],
        help='Max 10,000 rows. Required columns: category, payment_method, '
             'unit_price, discount_pct, delivery_delay_days, seller_rating, '
             'image_count, account_age_days, fulfilment_type',
        label_visibility='collapsed'
    )

    if uploaded is None:
        # Show sample preview
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Expected CSV format (sample)</div>',
                    unsafe_allow_html=True)
        st.dataframe(TEMPLATE_DATA, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # ── Score ─────────────────────────────────────────────────────────────────
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f'Could not read CSV: {e}')
        return

    if len(df_raw) == 0:
        st.error('Uploaded file is empty.')
        return

    if len(df_raw) > 10000:
        st.warning('File has more than 10,000 rows. Only the first 10,000 will be scored.')
        df_raw = df_raw.head(10000)

    with st.spinner(f'Scoring {len(df_raw):,} orders...'):
        scored_df, missing_cols = score_batch(
            df_raw, model, MODEL_FEATURES, ref_features
        )

    if missing_cols:
        st.error(f'Missing required columns: **{", ".join(missing_cols)}**')
        st.info('Download the template above to see the correct column names.')
        return

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    total       = len(scored_df)
    high_risk   = int((scored_df['risk_tier'] == 'High Risk').sum())
    medium_risk = int((scored_df['risk_tier'] == 'Medium Risk').sum())
    low_risk    = int((scored_df['risk_tier'] == 'Low Risk').sum())
    avg_prob    = scored_df['return_probability'].mean()
    est_savings = high_risk * 750 * 0.25  # 25% reduction if intervened

    st.markdown('<div class="section-title">Batch scoring results</div>',
                unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""<div class="kpi-card" style="border-top-color:#2874F0">
          <div class="kpi-label">Orders Scored</div>
          <div class="kpi-value">{total:,}</div>
          <div class="kpi-delta delta-neutral">Batch processed</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card" style="border-top-color:#DC2626">
          <div class="kpi-label">High Risk</div>
          <div class="kpi-value" style="color:#DC2626">{high_risk:,}</div>
          <div class="kpi-delta delta-up">Require intervention</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card" style="border-top-color:#D97706">
          <div class="kpi-label">Medium Risk</div>
          <div class="kpi-value" style="color:#D97706">{medium_risk:,}</div>
          <div class="kpi-delta delta-neutral">Monitor closely</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card" style="border-top-color:#15803D">
          <div class="kpi-label">Low Risk</div>
          <div class="kpi-value" style="color:#15803D">{low_risk:,}</div>
          <div class="kpi-delta delta-down">Standard processing</div>
        </div>""", unsafe_allow_html=True)
    with k5:
        st.markdown(f"""<div class="kpi-card" style="border-top-color:#15803D">
          <div class="kpi-label">Est. Savings if Actioned</div>
          <div class="kpi-value">₹{est_savings:,.0f}</div>
          <div class="kpi-delta delta-down">From high-risk intervention</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="chart-card"><div class="chart-title">'
                    'Return probability distribution</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=scored_df['return_probability'],
            nbinsx=30,
            marker_color='#2874F0',
            opacity=0.8,
            name='All orders'
        ))
        fig.add_vline(x=30, line_dash='dot', line_color='#D97706',
                      annotation_text='Medium threshold (30%)',
                      annotation_font_size=10)
        fig.add_vline(x=60, line_dash='dot', line_color='#DC2626',
                      annotation_text='High threshold (60%)',
                      annotation_font_size=10)
        fig.update_layout(**{**CHART_LAYOUT, 'height': 260,
                             'xaxis_title': 'Return Probability (%)',
                             'yaxis_title': 'Order Count',
                             'showlegend': False})
        st.plotly_chart(fig, use_container_width=True,
                        config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card"><div class="chart-title">'
                    'Risk tier breakdown</div>', unsafe_allow_html=True)
        tier_counts = scored_df['risk_tier'].value_counts().reset_index()
        tier_counts.columns = ['tier', 'count']
        color_map = {
            'High Risk': '#DC2626',
            'Medium Risk': '#D97706',
            'Low Risk': '#15803D'
        }
        fig2 = go.Figure(go.Bar(
            x=tier_counts['tier'],
            y=tier_counts['count'],
            marker_color=[color_map.get(t, '#9CA3AF') for t in tier_counts['tier']],
            text=tier_counts['count'],
            textposition='outside',
            textfont=dict(size=12, color='#374151')
        ))
        fig2.update_layout(**{**CHART_LAYOUT, 'height': 260,
                              'xaxis_title': '', 'yaxis_title': 'Orders',
                              'showlegend': False})
        st.plotly_chart(fig2, use_container_width=True,
                        config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Top risk factors chart
    st.markdown('<div class="chart-card"><div class="chart-title">'
                'Top risk factors across batch</div>', unsafe_allow_html=True)
    factor_counts = scored_df['top_risk_factor'].value_counts().reset_index()
    factor_counts.columns = ['factor', 'count']
    fig3 = go.Figure(go.Bar(
        x=factor_counts['count'],
        y=factor_counts['factor'],
        orientation='h',
        marker_color='#2874F0',
        text=factor_counts['count'],
        textposition='outside',
        textfont=dict(size=10)
    ))
    fig3.update_layout(**{**CHART_LAYOUT, 'height': 220,
                          'xaxis_title': 'Number of orders',
                          'yaxis_title': '', 'showlegend': False,
                          'margin': dict(l=0, r=60, t=10, b=0)})
    st.plotly_chart(fig3, use_container_width=True,
                    config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Scored table ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Scored order list '
                '(sorted by return probability — highest first)</div>',
                unsafe_allow_html=True)

    # Filter controls
    f1, f2, f3 = st.columns(3)
    with f1:
        tier_filter = st.selectbox(
            'Filter by risk tier',
            ['All', 'High Risk', 'Medium Risk', 'Low Risk'],
            label_visibility='visible'
        )
    with f2:
        min_prob = st.slider('Minimum probability (%)', 0, 100, 0)
    with f3:
        st.markdown('<br>', unsafe_allow_html=True)
        show_n = st.selectbox('Show top N rows', [25, 50, 100, 250, 'All'])

    display_df = scored_df.copy()
    if tier_filter != 'All':
        display_df = display_df[display_df['risk_tier'] == tier_filter]
    display_df = display_df[display_df['return_probability'] >= min_prob]
    if show_n != 'All':
        display_df = display_df.head(int(show_n))

    # Build HTML table
    def tier_badge(t):
        cls = {'High Risk': 'badge-high',
               'Medium Risk': 'badge-med',
               'Low Risk': 'badge-low'}.get(t, '')
        return f'<span class="badge {cls}">{t}</span>'

    display_cols = ['category', 'payment_method', 'unit_price',
                    'discount_pct', 'delivery_delay_days', 'seller_rating',
                    'return_probability', 'risk_tier',
                    'top_risk_factor', 'recommended_action']
    display_cols = [c for c in display_cols if c in display_df.columns]

    rows_html = ''
    for _, row in display_df[display_cols].iterrows():
        prob_color = ('#DC2626' if row['return_probability'] >= 60
                      else '#D97706' if row['return_probability'] >= 30
                      else '#15803D')
        rows_html += '<tr>'
        for col in display_cols:
            val = row[col]
            if col == 'risk_tier':
                rows_html += f'<td>{tier_badge(val)}</td>'
            elif col == 'return_probability':
                rows_html += (f'<td><strong style="color:{prob_color}">'
                              f'{val}%</strong></td>')
            elif col == 'unit_price':
                rows_html += f'<td>₹{float(val):,.0f}</td>'
            elif col == 'discount_pct':
                rows_html += f'<td>{val}%</td>'
            else:
                rows_html += f'<td>{val}</td>'
        rows_html += '</tr>'

    header_labels = {
        'category': 'Category',
        'payment_method': 'Payment',
        'unit_price': 'Price',
        'discount_pct': 'Discount',
        'delivery_delay_days': 'Delay (days)',
        'seller_rating': 'Seller Rating',
        'return_probability': 'Return Prob.',
        'risk_tier': 'Risk Tier',
        'top_risk_factor': 'Top Risk Factor',
        'recommended_action': 'Recommended Action',
    }
    headers = ''.join(
        f'<th>{header_labels.get(c, c)}</th>' for c in display_cols
    )

    st.markdown(f"""
    <div class="chart-card" style="overflow-x:auto">
    <table class="styled-table">
      <thead><tr>{headers}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>""", unsafe_allow_html=True)

    # ── Download results ──────────────────────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    dl_df = scored_df[display_cols].copy()
    csv_out = dl_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='⬇ Download scored results as CSV',
        data=csv_out,
        file_name='scored_orders.csv',
        mime='text/csv',
        use_container_width=False
    )

    if high_risk > 0:
        st.markdown(f"""<div class="warn-box">
        <strong>Action required:</strong> {high_risk:,} orders flagged as High Risk
        (return probability ≥ 60%). Recommended: require prepaid payment or add a
        pre-dispatch quality check for these orders. Estimated savings if actioned:
        <strong>₹{est_savings:,.0f}</strong>.
        </div>""", unsafe_allow_html=True)
