"""Overview — operational command center."""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from components import (page_header, metric_card, section_label, chart_container,
                        format_inr, callout_box, risk_badge, styled_plotly, COLORS)


def render(features, sellers, returns_df, sellers_info):
    page_header("Operations Overview",
                "Portfolio-level return intelligence and risk summary")

    total_orders = len(features)
    total_returns = int(features['is_returned'].sum())
    return_rate = features['is_returned'].mean() * 100
    avg_return_cost = returns_df['return_cost'].mean()
    total_return_cost = returns_df['return_cost'].sum()
    high_risk_orders = int((features['return_probability'] >= 0.6).sum())
    gmv_at_risk = features[features['return_probability'] >= 0.6]['final_price'].sum()
    high_risk_sellers = int((sellers['risk_tier'] == 'High Risk').sum())
    med_risk_sellers = int((sellers['risk_tier'] == 'Medium Risk').sum())

    # ── KPI row ───────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        metric_card("Total Orders", f"{total_orders:,}",
                     "FY 2024 dataset", border_color=COLORS['primary'])
    with k2:
        metric_card("Return Rate", f"{return_rate:.1f}%",
                     f"{total_returns:,} returns", delta_type='up',
                     border_color=COLORS['risk_high'])
    with k3:
        metric_card("Total Return Cost", format_inr(total_return_cost),
                     f"Avg {format_inr(avg_return_cost)} per return", delta_type='up',
                     border_color=COLORS['risk_high'])
    with k4:
        metric_card("GMV at Risk", format_inr(gmv_at_risk),
                     f"{high_risk_orders:,} high-risk orders", delta_type='up',
                     border_color=COLORS['risk_med'])
    with k5:
        metric_card("Flagged Sellers", f"{high_risk_sellers + med_risk_sellers:,}",
                     f"{high_risk_sellers} high / {med_risk_sellers} medium",
                     delta_type='up', border_color=COLORS['risk_med'])

    st.markdown("")

    # ── Needs Attention ───────────────────────────────────────────────────
    section_label("Needs Attention")
    na1, na2 = st.columns(2)

    with na1:
        merged = sellers.merge(
            sellers_info[['seller_id', 'seller_name', 'city']], on='seller_id', how='left')
        top5 = merged.nlargest(5, 'risk_score')[
            ['seller_name', 'risk_score', 'return_rate_pct', 'risk_tier']
        ].reset_index(drop=True)
        top5.index += 1

        rows = ""
        for _, r in top5.iterrows():
            rows += (f"<tr><td>{r['seller_name']}</td>"
                     f"<td>{r['return_rate_pct']:.1f}%</td>"
                     f"<td>{r['risk_score']:.0f}</td>"
                     f"<td>{risk_badge(r['risk_tier'])}</td></tr>")
        st.markdown(f"""
        <div style="font-size:12px;font-weight:500;color:#202124;margin-bottom:6px">
          Top risk sellers requiring review</div>
        <table style="width:100%;border-collapse:collapse;font-size:12px">
          <thead><tr style="border-bottom:1px solid #dadce0">
            <th style="text-align:left;padding:6px 8px;color:#5f6368;font-weight:500">Seller</th>
            <th style="text-align:left;padding:6px 8px;color:#5f6368;font-weight:500">Return %</th>
            <th style="text-align:left;padding:6px 8px;color:#5f6368;font-weight:500">Score</th>
            <th style="text-align:left;padding:6px 8px;color:#5f6368;font-weight:500">Tier</th>
          </tr></thead><tbody>{rows}</tbody>
        </table>""", unsafe_allow_html=True)

    with na2:
        cat = features.groupby('category').agg(
            total=('is_returned', 'count'), returned=('is_returned', 'sum')).reset_index()
        cat['rate'] = (cat['returned'] / cat['total'] * 100).round(1)
        cat = cat.sort_values('rate', ascending=False).head(5).reset_index(drop=True)
        cat.index += 1

        rows2 = ""
        for _, r in cat.iterrows():
            clr = '#d93025' if r['rate'] > 30 else '#e8710a' if r['rate'] > 20 else '#188038'
            rows2 += (f"<tr><td style='padding:6px 8px'>{r['category']}</td>"
                      f"<td style='padding:6px 8px'>{r['total']:,}</td>"
                      f"<td style='padding:6px 8px;color:{clr};font-weight:500'>{r['rate']}%</td></tr>")
        st.markdown(f"""
        <div style="font-size:12px;font-weight:500;color:#202124;margin-bottom:6px">
          Categories by return rate</div>
        <table style="width:100%;border-collapse:collapse;font-size:12px">
          <thead><tr style="border-bottom:1px solid #dadce0">
            <th style="text-align:left;padding:6px 8px;color:#5f6368;font-weight:500">Category</th>
            <th style="text-align:left;padding:6px 8px;color:#5f6368;font-weight:500">Orders</th>
            <th style="text-align:left;padding:6px 8px;color:#5f6368;font-weight:500">Return Rate</th>
          </tr></thead><tbody>{rows2}</tbody>
        </table>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Charts row 1 ──────────────────────────────────────────────────────
    section_label("Return Analysis")
    c1, c2 = st.columns(2)

    with c1:
        chart_container("Return rate by category")
        cat_all = features.groupby('category').agg(
            t=('is_returned', 'count'), r=('is_returned', 'sum')).reset_index()
        cat_all['rate'] = (cat_all['r'] / cat_all['t'] * 100).round(2)
        cat_all = cat_all.sort_values('rate', ascending=True)
        clrs = ['#d93025' if x > 30 else '#e8710a' if x > 15 else '#188038' for x in cat_all['rate']]
        fig = go.Figure(go.Bar(x=cat_all['rate'], y=cat_all['category'], orientation='h',
                               marker_color=clrs,
                               text=[f'{r}%' for r in cat_all['rate']],
                               textposition='outside', textfont=dict(size=10)))
        fig.add_vline(x=return_rate, line_dash='dot', line_color='#9CA3AF',
                      annotation_text=f'Avg {return_rate:.1f}%',
                      annotation_font_size=10, annotation_font_color='#5f6368')
        styled_plotly(fig, height=280, xaxis_title='Return Rate (%)', yaxis_title='',
                      margin=dict(l=0, r=60, t=10, b=0))

    with c2:
        chart_container("Return rate by payment method")
        pay = features.groupby('payment_method').agg(
            t=('is_returned', 'count'), r=('is_returned', 'sum')).reset_index()
        pay['rate'] = (pay['r'] / pay['t'] * 100).round(2)
        pay = pay.sort_values('rate', ascending=False)
        pclrs = ['#d93025' if m == 'COD' else '#1a73e8' for m in pay['payment_method']]
        fig2 = go.Figure(go.Bar(x=pay['payment_method'], y=pay['rate'],
                                marker_color=pclrs,
                                text=[f'{r}%' for r in pay['rate']],
                                textposition='outside', textfont=dict(size=10)))
        styled_plotly(fig2, height=280, xaxis_title='', yaxis_title='Return rate (%)')

    # ── Charts row 2 ──────────────────────────────────────────────────────
    c3, c4 = st.columns(2)
    with c3:
        chart_container("Return rate by delivery delay")
        delay = features.groupby('delay_bucket', observed=True).agg(
            t=('is_returned', 'count'), r=('is_returned', 'sum')).reset_index()
        delay['rate'] = (delay['r'] / delay['t'] * 100).round(2)
        fig3 = go.Figure(go.Scatter(
            x=delay['delay_bucket'].astype(str), y=delay['rate'],
            mode='lines+markers+text',
            line=dict(color='#d93025', width=2),
            marker=dict(size=7, color='#d93025', line=dict(color='#fff', width=2)),
            text=[f'{r}%' for r in delay['rate']],
            textposition='top center', textfont=dict(size=10),
            fill='tozeroy', fillcolor='rgba(217,48,37,0.05)'))
        styled_plotly(fig3, height=280, xaxis_title='Delivery delay', yaxis_title='Return rate (%)')

    with c4:
        chart_container("Return rate by seller rating")
        rat = features.groupby('rating_bucket', observed=True).agg(
            t=('is_returned', 'count'), r=('is_returned', 'sum')).reset_index()
        rat['rate'] = (rat['r'] / rat['t'] * 100).round(2)
        fig4 = go.Figure(go.Scatter(
            x=rat['rating_bucket'].astype(str), y=rat['rate'],
            mode='lines+markers+text',
            line=dict(color='#1a73e8', width=2),
            marker=dict(size=7, color='#1a73e8', line=dict(color='#fff', width=2)),
            text=[f'{r}%' for r in rat['rate']],
            textposition='top center', textfont=dict(size=10),
            fill='tozeroy', fillcolor='rgba(26,115,232,0.05)'))
        styled_plotly(fig4, height=280, xaxis_title='Seller rating', yaxis_title='Return rate (%)')

    # ── Return reasons ────────────────────────────────────────────────────
    section_label("Return Reasons")
    reasons = returns_df['return_reason'].value_counts().reset_index()
    reasons.columns = ['reason', 'count']
    reasons['pct'] = (reasons['count'] / reasons['count'].sum() * 100).round(1)
    rc1, rc2 = st.columns([1.3, 1])
    with rc1:
        fig_r = go.Figure(go.Bar(
            y=reasons['reason'][::-1], x=reasons['count'][::-1], orientation='h',
            marker_color='#1a73e8',
            text=[f"{c:,} ({p}%)" for c, p in zip(reasons['count'][::-1], reasons['pct'][::-1])],
            textposition='outside', textfont=dict(size=10)))
        styled_plotly(fig_r, height=300, xaxis_title='Count', yaxis_title='',
                      margin=dict(l=0, r=80, t=10, b=0))
    with rc2:
        top_reason = reasons.iloc[0]
        callout_box(
            f"<strong>Top return reason:</strong> {top_reason['reason']} "
            f"({top_reason['pct']}% of all returns). "
            f"Improving product descriptions and images could reduce this category.")
        resolution = returns_df['resolution'].value_counts()
        for res, cnt in resolution.items():
            pct = cnt / resolution.sum() * 100
            st.markdown(f"<div style='font-size:12px;color:#5f6368;padding:2px 0'>"
                        f"<strong>{res}:</strong> {cnt:,} ({pct:.1f}%)</div>",
                        unsafe_allow_html=True)

    # ── Key insights ──────────────────────────────────────────────────────
    section_label("Key Findings")
    ff_rate = features[features['fulfilment_type'] == 'Platform Fulfilled']['is_returned'].mean() * 100
    sf_rate = features[features['fulfilment_type'] == 'Seller Fulfilled']['is_returned'].mean() * 100
    cod_rate = features[features['payment_method'] == 'COD']['is_returned'].mean() * 100
    non_cod_rate = features[features['payment_method'] != 'COD']['is_returned'].mean() * 100

    i1, i2, i3 = st.columns(3)
    with i1:
        callout_box(f"<strong>COD drives highest returns.</strong> {cod_rate:.1f}% return rate "
                    f"vs {non_cod_rate:.1f}% for digital payments. Requiring prepaid for "
                    f"flagged COD orders could significantly reduce return costs.")
    with i2:
        callout_box(f"<strong>Seller quality is the biggest lever.</strong> Sellers rated below "
                    f"2.5 have significantly higher return rates. Seller Fulfilled ({sf_rate:.1f}%) "
                    f"vs Platform Fulfilled ({ff_rate:.1f}%).")
    with i3:
        callout_box("<strong>Delivery delays compound risk.</strong> 5+ day delays push return "
                    "rate well above the baseline. Enforcing delivery SLA could prevent "
                    "a meaningful share of return costs.")
