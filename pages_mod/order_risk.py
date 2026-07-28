"""Order Risk Analysis — manual prediction workspace."""
import streamlit as st
import pandas as pd
import numpy as np
from components import (page_header, section_label, callout_box, styled_plotly, COLORS)


def render(features, model, MODEL_FEATURES):
    page_header("Order Risk Analysis",
                "Enter order details to assess return probability and risk factors")

    cat_map = {c: i for i, c in enumerate(sorted(features['category'].unique()))}
    pay_map = {c: i for i, c in enumerate(sorted(features['payment_method'].unique()))}
    seg_map = {c: i for i, c in enumerate(sorted(features['customer_segment'].unique()))}
    ful_map = {c: i for i, c in enumerate(sorted(features['fulfilment_type'].unique()))}

    form_col, result_col = st.columns([1, 1.1])

    with form_col:
        # ── Order details ─────────────────────────────────────────────────
        section_label("Order Details")
        category = st.selectbox('Product category',
                                ['Electronics', 'Fashion', 'Home & Kitchen', 'Sports',
                                 'Toys', 'Beauty', 'Books', 'Grocery'])
        payment = st.selectbox('Payment method',
                               ['COD', 'UPI', 'Credit Card', 'Debit Card', 'Net Banking', 'Wallet'])
        oc1, oc2 = st.columns(2)
        with oc1:
            price = st.number_input('Price (₹)', min_value=50, max_value=200000, value=8000, step=500)
        with oc2:
            discount = st.slider('Discount %', 0, 60, 10)

        # ── Customer profile ──────────────────────────────────────────────
        section_label("Customer Profile")
        cc1, cc2 = st.columns(2)
        with cc1:
            acc_age = st.slider('Account age (days)', 1, 1000, 180)
        with cc2:
            st.markdown(f"""<div style="font-size:12px;color:#5f6368;margin-top:28px">
              {"🆕 New customer" if acc_age < 30 else "Returning customer"}
              · Est. orders: {max(1, acc_age // 30)}</div>""", unsafe_allow_html=True)

        # ── Seller & Fulfilment ───────────────────────────────────────────
        section_label("Seller & Fulfilment")
        sc1, sc2 = st.columns(2)
        with sc1:
            rating = st.slider('Seller rating', 1.0, 5.0, 4.0, step=0.1)
        with sc2:
            images = st.slider('Product images', 1, 10, 5)
        sc3, sc4 = st.columns(2)
        with sc3:
            delay = st.slider('Delivery delay (days)', 0, 10, 0)
        with sc4:
            fulfil = st.selectbox('Fulfilment type',
                                  ['Platform Fulfilled', 'Seller Fulfilled'])

    # ── Model inference ───────────────────────────────────────────────────
    avg_cr = features['cust_return_rate'].mean()
    avg_pr = features[features['category'] == category]['prod_return_rate'].mean()
    avg_sr = features[features['rating'].between(rating - 0.3, rating + 0.3)]['seller_return_rate'].mean()
    if np.isnan(avg_pr): avg_pr = 0.22
    if np.isnan(avg_sr): avg_sr = 0.25

    inp = {
        'unit_price': price, 'discount_pct': discount, 'quantity': 1,
        'final_price': price * (1 - discount / 100), 'delivery_delay_days': delay,
        'is_cod': int(payment == 'COD'),
        'is_flipkart_fulfilled': int(fulfil == 'Platform Fulfilled'),
        'avg_rating': 3.9, 'image_count': images,
        'price_bucket': min(5, max(1, int(price / 30000 * 4) + 1)),
        'account_age_days': acc_age, 'is_new_customer': int(acc_age < 30),
        'cust_total_orders': max(1, acc_age // 30),
        'cust_return_rate': avg_cr, 'cust_cod_ratio': 1.0 if payment == 'COD' else 0.2,
        'rating': rating, 'seller_return_rate': avg_sr, 'seller_avg_discount': discount,
        'prod_return_rate': avg_pr, 'prod_avg_discount_given': discount,
        'is_high_discount': int(discount >= 40), 'is_delayed': int(delay > 0),
        'is_low_rated_seller': int(rating < 3.5), 'is_low_image_count': int(images <= 2),
        'is_high_price': int(price > 20000),
        'category_enc': cat_map.get(category, 0),
        'customer_segment_enc': seg_map.get('Regular', 0),
        'payment_method_enc': pay_map.get(payment, 0),
        'fulfilment_enc': ful_map.get(fulfil, 0),
    }
    X_in = pd.DataFrame([inp])[MODEL_FEATURES].fillna(0)
    prob = float(model.predict_proba(X_in)[0][1])
    pct = round(prob * 100, 1)

    if prob >= 0.6:
        color, tier = '#d93025', 'High Risk'
        badge_bg, badge_fg = '#fce8e6', '#c5221f'
    elif prob >= 0.3:
        color, tier = '#e8710a', 'Medium Risk'
        badge_bg, badge_fg = '#fef7e0', '#e8710a'
    else:
        color, tier = '#188038', 'Low Risk'
        badge_bg, badge_fg = '#e6f4ea', '#188038'

    with result_col:
        # ── Model Output ──────────────────────────────────────────────────
        section_label("Model Output")
        st.markdown(f"""
        <div style="background:#fff;border:1px solid #dadce0;border-radius:4px;padding:20px;
             text-align:center;margin-bottom:12px">
          <div style="font-size:11px;color:#5f6368;text-transform:uppercase;
               letter-spacing:.04em;margin-bottom:8px">Return Probability</div>
          <div style="font-size:40px;font-weight:500;color:{color};line-height:1">{pct}%</div>
          <div style="background:#f1f3f4;border-radius:3px;height:6px;margin:12px auto;
               max-width:260px">
            <div style="background:{color};height:6px;border-radius:3px;
                 width:{min(pct, 100)}%;transition:width .3s"></div>
          </div>
          <span style="background:{badge_bg};color:{badge_fg};padding:4px 14px;
                border-radius:3px;font-size:12px;font-weight:500">{tier}</span>
        </div>""", unsafe_allow_html=True)

        # ── Recommended Action ────────────────────────────────────────────
        section_label("Recommended Action")
        if prob >= 0.6:
            callout_box("<strong>Require prepaid payment.</strong> Add quality check before "
                        "dispatch. Consider seller warning if pattern persists.", style='critical')
        elif prob >= 0.3:
            callout_box("<strong>Flag for post-delivery follow-up.</strong> Ensure delivery SLA "
                        "is met. No blocking action needed.", style='warning')
        else:
            callout_box("<strong>Standard processing.</strong> No intervention needed. Order "
                        "appears low-risk across all signals.")

        # ── Risk Drivers ──────────────────────────────────────────────────
        section_label("Risk Drivers")
        factors = []
        if payment == 'COD':
            factors.append(('COD payment', 'critical',
                            'COD has ~31% return rate vs ~23% for digital payments'))
        if rating < 3.5:
            factors.append(('Low seller rating', 'critical',
                            f'Seller rated {rating:.1f} — low-rated sellers have significantly higher return rate'))
        if delay >= 3:
            factors.append(('Delivery delay', 'critical',
                            f'{delay}-day delay — 3+ day delays substantially increase return risk'))
        if discount >= 40:
            factors.append(('Heavy discount', 'warning',
                            f'{discount}% discount — impulse buy risk increases returns'))
        if images <= 2:
            factors.append(('Low image count', 'warning',
                            f'Only {images} images — expectation mismatch risk'))
        if price > 20000:
            factors.append(('High price item', 'warning',
                            f'₹{price:,} — high-value items face greater return scrutiny'))
        if acc_age < 30:
            factors.append(('New customer', 'warning',
                            f'{acc_age}-day-old account — new customers return more frequently'))
        if category in ['Electronics', 'Fashion']:
            cat_rate = features[features['category'] == category]['is_returned'].mean() * 100
            factors.append((f'{category} category', 'info',
                            f'Category avg return rate: {cat_rate:.1f}%'))
        if not factors:
            factors.append(('No significant risk', 'info',
                            'Order appears low-risk across all signals'))

        for name, severity, desc in factors[:6]:
            border = {'critical': '#d93025', 'warning': '#e8710a', 'info': '#1a73e8'}[severity]
            st.markdown(f"""
            <div style="background:#fff;border:1px solid #dadce0;border-left:3px solid {border};
                 border-radius:0 4px 4px 0;padding:8px 12px;margin:4px 0;font-size:12px;
                 color:#3c4043">
              <strong style="color:#202124">{name}</strong> — {desc}
            </div>""", unsafe_allow_html=True)
