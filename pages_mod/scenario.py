"""Scenario Planner — intervention simulator with ROI analysis."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from components import (page_header, metric_card, section_label, chart_container,
                        format_inr, callout_box, styled_plotly, COLORS)


def render(features, returns_df):
    page_header("Scenario Planner",
                "Toggle interventions and see projected savings update in real time")

    avg_return_cost = returns_df['return_cost'].mean()
    total_return_cost = returns_df['return_cost'].sum()

    # Actual affected return counts from data
    cod_affected = int(((features['is_cod'] == 1) & (features['is_returned'] == 1)).sum())
    low_rating_affected = int(((features['rating'] < 3.5) & (features['is_returned'] == 1)).sum())
    delayed_affected = int(((features['delivery_delay_days'] >= 3) & (features['is_returned'] == 1)).sum())
    low_img_affected = int(((features['image_count'] <= 2) & (features['is_returned'] == 1)).sum())

    callout_box("Scenario results are <strong>estimates based on user-defined assumptions</strong>. "
                "Actual outcomes will depend on implementation details, market conditions, and "
                "customer behaviour. Use these projections to prioritize interventions, not as "
                "guaranteed forecasts.", style='warning')

    st.markdown("")

    ctrl_col, result_col = st.columns([1, 1.5])

    with ctrl_col:
        section_label("Intervention Controls")

        # 1 — COD
        with st.container(border=True):
            st.markdown("**1. COD Payment Enforcement**")
            cod_on = st.toggle('Require prepaid for high-risk COD orders', value=True, key='cod_t')
            cod_pct = st.slider('Expected return reduction %', 5, 50, 25, key='cod_s',
                                disabled=not cod_on,
                                help="Percentage of affected COD returns you expect to prevent")
            st.caption(f"{cod_affected:,} COD returns affected · "
                       f"Avg cost {format_inr(avg_return_cost)}/return")

        # 2 — Seller quality
        with st.container(border=True):
            st.markdown("**2. Seller Quality Control**")
            sel_on = st.toggle('Warn/suspend sellers rated < 3.5 ★', value=True, key='sel_t')
            sel_pct = st.slider('Expected return reduction %', 5, 50, 30, key='sel_s',
                                disabled=not sel_on)
            st.caption(f"{low_rating_affected:,} low-rating returns affected")

        # 3 — Delivery SLA
        with st.container(border=True):
            st.markdown("**3. Delivery SLA Enforcement**")
            del_on = st.toggle('Penalise sellers with 3+ day delays', value=True, key='del_t')
            del_pct = st.slider('Expected return reduction %', 5, 60, 40, key='del_s',
                                disabled=not del_on)
            st.caption(f"{delayed_affected:,} delayed-delivery returns affected")

        # 4 — Image policy
        with st.container(border=True):
            st.markdown("**4. Product Image Requirements**")
            img_on = st.toggle('Require 5+ images for all listings', value=False, key='img_t')
            img_pct = st.slider('Expected return reduction %', 5, 40, 20, key='img_s',
                                disabled=not img_on)
            st.caption(f"{low_img_affected:,} low-image returns affected")

    # ── Calculate savings ─────────────────────────────────────────────────
    cod_save = int(cod_affected * (cod_pct / 100) * avg_return_cost) if cod_on else 0
    sel_save = int(low_rating_affected * (sel_pct / 100) * avg_return_cost) if sel_on else 0
    del_save = int(delayed_affected * (del_pct / 100) * avg_return_cost) if del_on else 0
    img_save = int(low_img_affected * (img_pct / 100) * avg_return_cost) if img_on else 0
    total_save = cod_save + sel_save + del_save + img_save
    pct_saved = total_save / total_return_cost * 100 if total_return_cost > 0 else 0

    # Implementation costs (one-time estimates)
    impl_costs = {
        'COD Enforcement': 50_000,
        'Seller Quality': 150_000,
        'Delivery SLA': 75_000,
        'Image Policy': 30_000,
    }

    with result_col:
        # ── Scenario summary ──────────────────────────────────────────────
        section_label("Scenario Summary")

        s1, s2, s3 = st.columns(3)
        with s1:
            metric_card("Total Projected Savings", format_inr(total_save),
                         f"{pct_saved:.1f}% of total return cost",
                         delta_type='down', border_color=COLORS['risk_low'])
        with s2:
            active_count = sum([cod_on, sel_on, del_on, img_on])
            returns_prevented = int(
                (cod_affected * cod_pct / 100 if cod_on else 0) +
                (low_rating_affected * sel_pct / 100 if sel_on else 0) +
                (delayed_affected * del_pct / 100 if del_on else 0) +
                (low_img_affected * img_pct / 100 if img_on else 0))
            metric_card("Returns Prevented", f"{returns_prevented:,}",
                         f"{active_count} interventions active",
                         border_color=COLORS['primary'])
        with s3:
            total_impl = sum(impl_costs[k] for k, a in
                             [('COD Enforcement', cod_on), ('Seller Quality', sel_on),
                              ('Delivery SLA', del_on), ('Image Policy', img_on)] if a)
            overall_roi = ((total_save - total_impl) / total_impl * 100) if total_impl > 0 else 0
            metric_card("Overall ROI", f"{overall_roi:.0f}%",
                         f"Impl. cost: {format_inr(total_impl)}",
                         border_color=COLORS['risk_low'] if overall_roi > 0 else COLORS['risk_med'])

        st.markdown("")

        # ── Savings breakdown chart ───────────────────────────────────────
        if total_save > 0:
            chart_container("Savings breakdown")
            items = [('COD Enforcement', cod_save, '#1a73e8'),
                     ('Seller Quality', sel_save, '#d93025'),
                     ('Delivery SLA', del_save, '#e8710a'),
                     ('Image Policy', img_save, '#188038')]
            active = [(n, v, c) for n, v, c in items if v > 0]
            if active:
                fig = go.Figure(go.Bar(
                    x=[n for n, v, c in active],
                    y=[v / 100_000 for n, v, c in active],
                    marker_color=[c for n, v, c in active],
                    text=[f'₹{v / 100_000:.1f}L' for n, v, c in active],
                    textposition='outside', textfont=dict(size=11)))
                styled_plotly(fig, height=230, xaxis_title='', yaxis_title='₹ Lakhs')

        # ── Intervention detail table ─────────────────────────────────────
        section_label("Intervention Details")
        rows = []
        for name, affected, pct_r, save, active, impl_key in [
            ('COD Enforcement', cod_affected, cod_pct, cod_save, cod_on, 'COD Enforcement'),
            ('Seller Quality', low_rating_affected, sel_pct, sel_save, sel_on, 'Seller Quality'),
            ('Delivery SLA', delayed_affected, del_pct, del_save, del_on, 'Delivery SLA'),
            ('Image Policy', low_img_affected, img_pct, img_save, img_on, 'Image Policy'),
        ]:
            impl = impl_costs[impl_key]
            roi = ((save - impl) / impl * 100) if active and save > 0 else 0
            payback = (impl / (save / 12)) if active and save > 0 else 0
            rows.append({
                'Intervention': name,
                'Status': 'Active' if active else 'Off',
                'Returns Affected': affected,
                'Reduction %': f"{pct_r}%" if active else '—',
                'Savings': format_inr(save) if active else '—',
                'Impl. Cost': format_inr(impl),
                'ROI': f"{roi:.0f}%" if active and save > 0 else '—',
                'Payback': f"{payback:.1f} mo" if active and save > 0 else '—',
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Revenue at risk ───────────────────────────────────────────────
    st.markdown("")
    section_label("Revenue at Risk")
    risk_orders = features[features['return_probability'] >= 0.60]
    total_gmv_at_risk = risk_orders['final_price'].sum()
    high_risk_count = len(risk_orders)
    avg_order_val = risk_orders['final_price'].mean()
    pct_of_gmv = total_gmv_at_risk / features['final_price'].sum() * 100

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        metric_card("High Risk Orders", f"{high_risk_count:,}",
                     "prob ≥ 60%", delta_type='up', border_color=COLORS['risk_high'])
    with r2:
        metric_card("GMV at Risk", format_inr(total_gmv_at_risk),
                     "Total value of flagged orders", delta_type='up',
                     border_color=COLORS['risk_high'])
    with r3:
        metric_card("% of Total GMV", f"{pct_of_gmv:.1f}%",
                     "Revenue concentration", delta_type='up',
                     border_color=COLORS['risk_med'])
    with r4:
        metric_card("Avg Order Value", format_inr(avg_order_val),
                     "Per high-risk order", border_color=COLORS['risk_med'])
