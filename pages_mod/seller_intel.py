"""Seller Intelligence — seller management workspace."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from components import (page_header, metric_card, section_label, chart_container,
                        format_inr, callout_box, risk_badge, styled_plotly, COLORS)


def render(features, sellers, sellers_info, returns_df):
    page_header("Seller Intelligence",
                "Identify which sellers drive returns and quantify business impact")

    # ── Merge seller data ─────────────────────────────────────────────────
    merged = sellers.merge(
        sellers_info[['seller_id', 'seller_name', 'city', 'rating']],
        on='seller_id', how='left')

    # ── KPI row ───────────────────────────────────────────────────────────
    high_risk_s = int((sellers['risk_tier'] == 'High Risk').sum())
    med_risk_s = int((sellers['risk_tier'] == 'Medium Risk').sum())
    ff_rate = features[features['fulfilment_type'] == 'Platform Fulfilled']['is_returned'].mean() * 100
    sf_rate = features[features['fulfilment_type'] == 'Seller Fulfilled']['is_returned'].mean() * 100
    top100_rate = sellers.nlargest(100, 'risk_score')['actual_return_rate'].mean() * 100

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("High Risk Sellers", f"{high_risk_s}",
                     "Need immediate review", delta_type='up',
                     border_color=COLORS['risk_high'])
    with k2:
        metric_card("Top 100 Avg Return Rate", f"{top100_rate:.1f}%",
                     "Highest-risk sellers", delta_type='up',
                     border_color=COLORS['risk_high'])
    with k3:
        metric_card("Platform Fulfilled", f"{ff_rate:.1f}%",
                     "Return rate", delta_type='down',
                     border_color=COLORS['risk_low'])
    with k4:
        metric_card("Seller Fulfilled", f"{sf_rate:.1f}%",
                     "Return rate", delta_type='up',
                     border_color=COLORS['risk_med'])

    st.markdown("")

    # ── Filters ───────────────────────────────────────────────────────────
    section_label("Seller Directory")
    f1, f2, f3 = st.columns([2, 1, 1])
    with f1:
        search = st.text_input("Search seller name or ID", placeholder="Type to search...")
    with f2:
        tier_filter = st.selectbox("Risk tier", ["All", "High Risk", "Medium Risk", "Low Risk"])
    with f3:
        ful_filter = st.selectbox("Fulfilment", ["All", "Platform Fulfilled", "Seller Fulfilled"])

    # ── Apply filters ─────────────────────────────────────────────────────
    display = merged.copy()
    if search:
        mask = (display['seller_name'].str.contains(search, case=False, na=False) |
                display['seller_id'].str.contains(search, case=False, na=False))
        display = display[mask]
    if tier_filter != "All":
        display = display[display['risk_tier'] == tier_filter]
    if ful_filter != "All":
        display = display[display['fulfilment_type'] == ful_filter]

    display_sorted = display.sort_values('risk_score', ascending=False).reset_index(drop=True)

    # ── Recommended action column ─────────────────────────────────────────
    def get_action(row):
        if row['risk_score'] >= 70:
            return "Suspend / migrate to FF"
        elif row['risk_score'] >= 50:
            return "Performance warning"
        elif row['risk_tier'] == 'Medium Risk':
            return "Monitor closely"
        return "No action needed"

    display_sorted['action'] = display_sorted.apply(get_action, axis=1)

    # ── Build table ───────────────────────────────────────────────────────
    table_df = display_sorted[[
        'seller_id', 'seller_name', 'city', 'total_items_sold',
        'return_rate_pct', 'avg_seller_rating', 'risk_score',
        'total_return_cost', 'risk_tier', 'action'
    ]].rename(columns={
        'seller_id': 'ID', 'seller_name': 'Seller', 'city': 'City',
        'total_items_sold': 'Orders', 'return_rate_pct': 'Return %',
        'avg_seller_rating': 'Rating', 'risk_score': 'Risk Score',
        'total_return_cost': 'Return Cost (₹)', 'risk_tier': 'Tier',
        'action': 'Recommended Action'
    })

    st.markdown(f"<div style='font-size:12px;color:#5f6368;margin-bottom:6px'>"
                f"Showing {len(table_df):,} of {len(merged):,} sellers</div>",
                unsafe_allow_html=True)

    st.dataframe(
        table_df,
        use_container_width=True,
        height=400,
        column_config={
            "Return %": st.column_config.NumberColumn(format="%.1f%%"),
            "Rating": st.column_config.NumberColumn(format="%.1f ★"),
            "Risk Score": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.0f"),
            "Return Cost (₹)": st.column_config.NumberColumn(format="₹%.0f"),
        },
        hide_index=True,
    )

    # ── CSV export ────────────────────────────────────────────────────────
    csv = table_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download as CSV", csv, "seller_intelligence.csv",
                       "text/csv", use_container_width=False)

    # ── Seller detail expander ────────────────────────────────────────────
    if len(display_sorted) > 0:
        st.markdown("")
        section_label("Seller Detail")
        sel_options = display_sorted['seller_id'].head(50).tolist()
        selected_id = st.selectbox("Select a seller to inspect",
                                   sel_options, format_func=lambda x:
                                   f"{x} — {display_sorted[display_sorted['seller_id']==x]['seller_name'].values[0]}")
        sel_row = display_sorted[display_sorted['seller_id'] == selected_id].iloc[0]

        with st.expander(f"Details: {sel_row['seller_name']} ({selected_id})", expanded=True):
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.metric("Orders Sold", f"{sel_row['total_items_sold']:,}")
            with d2:
                st.metric("Return Rate", f"{sel_row['return_rate_pct']:.1f}%")
            with d3:
                st.metric("Risk Score", f"{sel_row['risk_score']:.0f}/100")
            with d4:
                st.metric("Rating", f"{sel_row['avg_seller_rating']:.1f} ★")

            st.markdown(f"**City:** {sel_row['city']} · "
                        f"**Fulfilment:** {sel_row['fulfilment_type']} · "
                        f"**Return Cost:** {format_inr(sel_row['total_return_cost'])} · "
                        f"**Avg Discount:** {sel_row['avg_discount']:.1f}%")

            if sel_row['risk_score'] >= 50:
                callout_box(f"<strong>Action required:</strong> {sel_row['action']}. "
                            f"This seller's return rate ({sel_row['return_rate_pct']:.1f}%) is "
                            f"above the platform average.", style='warning')

            # Products by this seller
            seller_orders = features[features['seller_id'] == selected_id]
            if len(seller_orders) > 0:
                cat_breakdown = seller_orders.groupby('category').agg(
                    orders=('is_returned', 'count'),
                    returns=('is_returned', 'sum')
                ).reset_index()
                cat_breakdown['rate'] = (cat_breakdown['returns'] / cat_breakdown['orders'] * 100).round(1)
                st.markdown("**Category breakdown:**")
                st.dataframe(cat_breakdown.sort_values('rate', ascending=False),
                             use_container_width=True, hide_index=True,
                             column_config={
                                 "rate": st.column_config.NumberColumn("Return %", format="%.1f%%")
                             })

    st.markdown("")

    # ── Charts ────────────────────────────────────────────────────────────
    section_label("Analysis")
    ch1, ch2 = st.columns(2)

    with ch1:
        chart_container("Seller risk tier distribution")
        tc = sellers['risk_tier'].value_counts().reset_index()
        tc.columns = ['tier', 'count']
        tc_colors = {'High Risk': '#d93025', 'Medium Risk': '#e8710a', 'Low Risk': '#188038'}
        fig = go.Figure(go.Bar(
            x=tc['tier'], y=tc['count'],
            marker_color=[tc_colors.get(t, '#9CA3AF') for t in tc['tier']],
            text=tc['count'], textposition='outside', textfont=dict(size=11)))
        styled_plotly(fig, height=260, xaxis_title='', yaxis_title='Number of sellers')

    with ch2:
        chart_container("Fulfilment type — return rate by category")
        fc = features.groupby(['category', 'fulfilment_type'])['is_returned'].mean().mul(100).round(2).reset_index()
        fig2 = px.bar(fc, x='category', y='is_returned', color='fulfilment_type',
                      barmode='group', text_auto='.1f',
                      color_discrete_map={'Platform Fulfilled': '#188038',
                                          'Seller Fulfilled': '#d93025'})
        styled_plotly(fig2, height=260, xaxis_title='', yaxis_title='Return rate (%)',
                      showlegend=True)

    # ── Monthly trend ─────────────────────────────────────────────────────
    chart_container("Return rate trend (monthly)")
    features_copy = features.copy()
    features_copy['order_date'] = pd.to_datetime(features_copy['order_date'])
    features_copy['month'] = features_copy['order_date'].dt.to_period('M').astype(str)

    monthly_ful = features_copy.groupby(['month', 'fulfilment_type']).agg(
        total=('is_returned', 'count'), returned=('is_returned', 'sum')).reset_index()
    monthly_ful['rate'] = (monthly_ful['returned'] / monthly_ful['total'] * 100).round(2)
    monthly_all = features_copy.groupby('month').agg(
        total=('is_returned', 'count'), returned=('is_returned', 'sum')).reset_index()
    monthly_all['rate'] = (monthly_all['returned'] / monthly_all['total'] * 100).round(2)

    ff_m = monthly_ful[monthly_ful['fulfilment_type'] == 'Platform Fulfilled']
    sf_m = monthly_ful[monthly_ful['fulfilment_type'] == 'Seller Fulfilled']

    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=monthly_all['month'], y=monthly_all['rate'],
                               name='Overall', mode='lines+markers',
                               line=dict(color='#202124', width=2),
                               marker=dict(size=6)))
    fig_t.add_trace(go.Scatter(x=ff_m['month'], y=ff_m['rate'],
                               name='Platform Fulfilled', mode='lines+markers',
                               line=dict(color='#188038', width=1.5, dash='dot'),
                               marker=dict(size=5)))
    fig_t.add_trace(go.Scatter(x=sf_m['month'], y=sf_m['rate'],
                               name='Seller Fulfilled', mode='lines+markers',
                               line=dict(color='#d93025', width=1.5, dash='dot'),
                               marker=dict(size=5)))
    styled_plotly(fig_t, height=280, xaxis_title='Month', yaxis_title='Return rate (%)',
                  showlegend=True)

    callout_box("<strong>Recommendation:</strong> Sellers with risk score above 70 should receive "
                "an automated performance warning. Sellers above 85 should be migrated from Seller "
                "Fulfilled to Platform Fulfilled or temporarily suspended.", style='warning')
