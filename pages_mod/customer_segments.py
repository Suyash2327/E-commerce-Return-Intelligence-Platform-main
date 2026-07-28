"""
pages_mod/customer_segments.py
Customer Return Behaviour Segmentation — RFM-style analysis for return abuse detection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from components import (
    page_header, metric_card, section_label, callout_box,
    styled_plotly, COLORS, format_inr
)


# ── Segmentation logic ────────────────────────────────────────────────────────
def build_customer_profiles(features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate order-level data to customer-level and assign a behaviour segment."""
    cust = features.groupby('customer_id').agg(
        total_orders    = ('order_id',          'count'),
        total_returns   = ('is_returned',        'sum'),
        return_rate     = ('is_returned',        'mean'),
        avg_order_value = ('final_price',        'mean'),
        total_gmv       = ('final_price',        'sum'),
        account_age     = ('account_age_days',   'first'),
        segment         = ('customer_segment',   'first'),
        cod_ratio       = ('cust_cod_ratio',     'first'),
        avg_return_prob = ('return_probability', 'mean'),
    ).reset_index()

    # ── Segment assignment rules ──────────────────────────────────────────────
    def assign_segment(row):
        rr  = row['return_rate']
        n   = row['total_orders']
        age = row['account_age']

        if rr >= 0.50 and n >= 3:
            return 'Serial Returner'
        if rr >= 1.0 or (rr >= 0.85 and n <= 3 and age < 90):
            return 'Ghost Buyer'
        if rr <= 0.10 and n >= 4:
            return 'Loyal Low-Risk'
        return 'Occasional Returner'

    cust['behaviour_segment'] = cust.apply(assign_segment, axis=1)
    cust['return_rate_pct']   = (cust['return_rate'] * 100).round(1)
    cust['avg_order_value']   = cust['avg_order_value'].round(0)
    cust['total_gmv']         = cust['total_gmv'].round(0)
    return cust


SEGMENT_META = {
    'Serial Returner': {
        'color':      COLORS['risk_high'],
        'bg':         '#fce8e6',
        'icon':       '🔴',
        'policy':     'Reduce return window to 7 days · Flag account for manual review',
        'policy_style': 'critical',
        'description': 'Return rate >50% across 3+ orders. These customers cost the most in logistics.',
    },
    'Ghost Buyer': {
        'color':      '#7b1fa2',
        'bg':         '#f3e5f5',
        'icon':       '👻',
        'policy':     'Require prepaid on all future orders · Block COD · Limit to 1 active return at a time',
        'policy_style': 'critical',
        'description': 'Returns almost everything, often new or short-tenure accounts. Likely abuse pattern.',
    },
    'Occasional Returner': {
        'color':      COLORS['risk_med'],
        'bg':         '#fef7e0',
        'icon':       '🟡',
        'policy':     'Standard return policy · Send size/fit guides for Fashion orders',
        'policy_style': 'warning',
        'description': 'Returns 1–2 items, usually size or expectation mismatch. Normal behaviour.',
    },
    'Loyal Low-Risk': {
        'color':      COLORS['risk_low'],
        'bg':         '#e6f4ea',
        'icon':       '🟢',
        'policy':     'Fast-track returns, no questions asked · Priority customer support',
        'policy_style': 'info',
        'description': 'High order count, very low return rate. Most valuable customer cohort.',
    },
}


# ── Main render ───────────────────────────────────────────────────────────────
def render(features: pd.DataFrame, returns_df: pd.DataFrame):
    page_header(
        "Customer Behaviour Segmentation",
        "RFM-style return abuse detection — identify serial returners, ghost buyers, and loyal customers"
    )

    with st.spinner("Building customer profiles from 41,000+ customers..."):
        cust = build_customer_profiles(features)

    seg_counts = cust['behaviour_segment'].value_counts()
    avg_return_cost = returns_df['return_cost'].mean() if 'return_cost' in returns_df.columns else 5271

    # ── KPI cards ─────────────────────────────────────────────────────────────
    section_label("Portfolio Overview")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        metric_card("Total Customers", f"{len(cust):,}", "Unique accounts")
    with k2:
        n = seg_counts.get('Serial Returner', 0)
        metric_card("Serial Returners", f"{n:,}",
                    f"{n/len(cust)*100:.1f}% of base",
                    delta_type='up', border_color=COLORS['risk_high'])
    with k3:
        n = seg_counts.get('Ghost Buyer', 0)
        metric_card("Ghost Buyers", f"{n:,}",
                    f"{n/len(cust)*100:.1f}% of base",
                    delta_type='up', border_color='#7b1fa2')
    with k4:
        n = seg_counts.get('Loyal Low-Risk', 0)
        metric_card("Loyal Low-Risk", f"{n:,}",
                    f"{n/len(cust)*100:.1f}% of base",
                    delta_type='down', border_color=COLORS['risk_low'])
    with k5:
        # Cost attribution: serial + ghost buyers
        high_abuse = cust[cust['behaviour_segment'].isin(['Serial Returner', 'Ghost Buyer'])]
        abuse_cost = (high_abuse['total_returns'].sum() * avg_return_cost)
        metric_card("Abuse-Driven Return Cost", format_inr(abuse_cost),
                    "Serial + Ghost buyers", border_color=COLORS['risk_high'])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Segment breakdown charts ───────────────────────────────────────────────
    section_label("Segment Analysis")
    ch1, ch2 = st.columns(2)

    with ch1:
        # Donut chart — segment distribution
        labels  = list(seg_counts.index)
        values  = list(seg_counts.values)
        colors  = [SEGMENT_META[s]['color'] for s in labels]
        fig_pie = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.55,
            marker=dict(colors=colors),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Customers: %{value:,}<br>Share: %{percent}<extra></extra>',
        ))
        fig_pie.update_traces(textfont_size=12)
        styled_plotly(fig_pie, height=280,
                      title='Customer Segment Distribution', showlegend=False)

    with ch2:
        # Avg return rate by segment
        seg_stats = cust.groupby('behaviour_segment').agg(
            avg_return_rate=('return_rate_pct', 'mean'),
            avg_orders=('total_orders', 'mean'),
            count=('customer_id', 'count'),
        ).reset_index().sort_values('avg_return_rate', ascending=True)

        bar_colors = [SEGMENT_META[s]['color'] for s in seg_stats['behaviour_segment']]
        fig_bar = go.Figure(go.Bar(
            x=seg_stats['avg_return_rate'],
            y=seg_stats['behaviour_segment'],
            orientation='h',
            marker_color=bar_colors,
            text=[f"{v:.1f}%" for v in seg_stats['avg_return_rate']],
            textposition='outside',
        ))
        styled_plotly(fig_bar, height=280,
                      xaxis_title='Avg Return Rate (%)', yaxis_title='',
                      showlegend=False, title='Avg Return Rate by Segment',
                      margin=dict(l=0, r=60, t=28, b=0))

    # Return rate scatter — orders vs return rate (sampled for perf)
    sample = cust.sample(min(3000, len(cust)), random_state=42)
    color_map = {s: SEGMENT_META[s]['color'] for s in SEGMENT_META}
    fig_scatter = px.scatter(
        sample,
        x='total_orders', y='return_rate_pct',
        color='behaviour_segment',
        color_discrete_map=color_map,
        hover_data=['customer_id', 'avg_order_value', 'total_gmv'],
        labels={
            'total_orders': 'Total Orders',
            'return_rate_pct': 'Return Rate (%)',
            'behaviour_segment': 'Segment',
        },
        opacity=0.6,
        size_max=8,
    )
    fig_scatter.update_traces(marker=dict(size=6))
    styled_plotly(fig_scatter, height=300,
                  title='Customer Map — Orders vs Return Rate (sample of 3,000)',
                  showlegend=True)

    # ── Segment cards with policy recommendations ─────────────────────────────
    section_label("Policy Recommendations by Segment")

    for seg_name, meta in SEGMENT_META.items():
        seg_data = cust[cust['behaviour_segment'] == seg_name]
        if seg_data.empty:
            continue

        count      = len(seg_data)
        avg_rr     = seg_data['return_rate_pct'].mean()
        avg_orders = seg_data['total_orders'].mean()
        avg_gmv    = seg_data['total_gmv'].mean()
        total_cost = seg_data['total_returns'].sum() * avg_return_cost

        with st.container(border=True):
            hdr_col, stat_col = st.columns([2, 3])
            with hdr_col:
                st.markdown(f"""
                <div style="padding:4px 0">
                  <div style="font-size:18px;font-weight:700;color:{meta['color']};margin-bottom:4px">
                    {meta['icon']} {seg_name}
                  </div>
                  <div style="font-size:13px;color:#5f6368">{meta['description']}</div>
                </div>""", unsafe_allow_html=True)

            with stat_col:
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    metric_card("Customers", f"{count:,}", border_color=meta['color'])
                with s2:
                    metric_card("Avg Return Rate", f"{avg_rr:.1f}%", border_color=meta['color'])
                with s3:
                    metric_card("Avg Orders", f"{avg_orders:.1f}", border_color=meta['color'])
                with s4:
                    metric_card("Est. Return Cost", format_inr(total_cost), border_color=meta['color'])

            st.markdown("<div style='margin-top:8px'>", unsafe_allow_html=True)
            callout_box(f"<strong>Recommended Policy:</strong> {meta['policy']}",
                        style=meta['policy_style'])
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # ── Drill-down table ──────────────────────────────────────────────────────
    section_label("Customer Lookup")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        seg_filter = st.selectbox(
            "Filter by segment",
            ['All'] + list(SEGMENT_META.keys())
        )
    with col_f2:
        min_orders = st.slider("Min. orders", 1, 20, 1)
    with col_f3:
        sort_by = st.selectbox("Sort by",
                               ['return_rate_pct', 'total_orders', 'total_returns', 'avg_order_value'])

    display = cust.copy()
    if seg_filter != 'All':
        display = display[display['behaviour_segment'] == seg_filter]
    display = display[display['total_orders'] >= min_orders]
    display = display.sort_values(sort_by, ascending=False).head(200)

    display_cols = {
        'customer_id':        'Customer ID',
        'behaviour_segment':  'Segment',
        'total_orders':       'Orders',
        'total_returns':      'Returns',
        'return_rate_pct':    'Return Rate (%)',
        'avg_order_value':    'Avg Order (₹)',
        'total_gmv':          'Total GMV (₹)',
        'account_age':        'Account Age (days)',
        'cod_ratio':          'COD Ratio',
    }
    display_renamed = display[list(display_cols.keys())].rename(columns=display_cols)
    st.dataframe(display_renamed, use_container_width=True, hide_index=True)

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    csv_out = cust.rename(columns={'return_rate_pct': 'return_rate_%'}).to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇  Export full customer segmentation as CSV",
        data=csv_out,
        file_name='customer_segments.csv',
        mime='text/csv',
    )

    callout_box(
        f"<strong>{seg_counts.get('Serial Returner', 0) + seg_counts.get('Ghost Buyer', 0):,} "
        f"high-risk customers</strong> (Serial Returners + Ghost Buyers) are responsible for an "
        f"estimated <strong>{format_inr((cust[cust['behaviour_segment'].isin(['Serial Returner','Ghost Buyer'])]['total_returns'].sum() * avg_return_cost))}</strong> "
        f"in return costs. Applying targeted policies to this cohort alone could recover significant margin.",
        style='warning'
    )
