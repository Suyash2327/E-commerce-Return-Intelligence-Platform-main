"""Reusable UI components for the Return Intelligence Platform."""
import streamlit as st
import plotly.graph_objects as go

# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    'primary': '#1a73e8',
    'risk_high': '#d93025',
    'risk_med': '#e8710a',
    'risk_low': '#188038',
    'text': '#202124',
    'text_secondary': '#5f6368',
    'border': '#dadce0',
    'surface': '#f8f9fa',
    'white': '#ffffff',
}

CHART_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=0, r=10, t=28, b=0),
    font=dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
              size=11, color='#5f6368'),
    xaxis=dict(gridcolor='#f1f3f4', linecolor='#dadce0', zeroline=False),
    yaxis=dict(gridcolor='#f1f3f4', linecolor='#dadce0', zeroline=False),
    showlegend=False,
    legend=dict(orientation='h', y=1.08, x=0, font=dict(size=11)),
    hoverlabel=dict(bgcolor='#202124', font_size=11, font_color='#fff'),
)

FEATURE_LABELS = {
    'cust_return_rate': 'Customer Return History',
    'prod_return_rate': 'Product Return History',
    'is_cod': 'Cash on Delivery',
    'discount_pct': 'Discount Percentage',
    'delivery_delay_days': 'Delivery Delay',
    'cust_cod_ratio': 'Customer COD Usage',
    'category_enc': 'Product Category',
    'prod_avg_discount_given': 'Product Avg Discount',
    'seller_return_rate': 'Seller Return History',
    'rating': 'Seller Rating',
    'account_age_days': 'Account Age',
    'final_price': 'Order Value',
    'payment_method_enc': 'Payment Method',
    'unit_price': 'Unit Price',
}


def page_header(title, subtitle, date_label="Jan–Dec 2024"):
    """Render a consistent page header with data freshness badge."""
    st.markdown(f"""
    <div style="margin-bottom:28px">
      <div style="display:flex;align-items:baseline;gap:16px;margin-bottom:4px">
        <h1 style="font-size:32px;font-weight:700;color:#202124;margin:0;padding:0;line-height:1.2">{title}</h1>
        <span style="font-size:12px;color:#5f6368;background:#f1f3f4;padding:4px 10px;
              border-radius:12px;font-weight:500">Data: {date_label}</span>
      </div>
      <div style="font-size:15px;color:#5f6368;margin-top:6px">{subtitle}</div>
    </div>""", unsafe_allow_html=True)


def metric_card(label, value, delta=None, delta_type='neutral', border_color=None):
    """Render a compact KPI card."""
    bc = border_color or '#dadce0'
    dc = {'up': '#d93025', 'down': '#188038', 'neutral': '#5f6368'}.get(delta_type, '#5f6368')
    delta_html = f'<div style="font-size:11px;color:{dc};margin-top:4px">{delta}</div>' if delta else ''
    st.markdown(f"""
    <div style="background:#fff;border:1px solid #dadce0;border-top:3px solid {bc};
         border-radius:4px;padding:14px 16px">
      <div style="font-size:11px;color:#5f6368;text-transform:uppercase;letter-spacing:.03em;
           margin-bottom:6px">{label}</div>
      <div style="font-size:22px;font-weight:500;color:#202124;line-height:1">{value}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)


def risk_badge(tier):
    """Return HTML for a risk tier badge."""
    styles = {
        'High Risk': ('background:#fce8e6;color:#c5221f', 'High'),
        'Medium Risk': ('background:#fef7e0;color:#e8710a', 'Medium'),
        'Low Risk': ('background:#e6f4ea;color:#188038', 'Low'),
    }
    style, label = styles.get(tier, ('background:#f1f3f4;color:#5f6368', tier))
    return f'<span style="{style};padding:2px 8px;border-radius:3px;font-size:11px;font-weight:500">{label}</span>'


def section_label(text):
    """Render a section divider label."""
    st.markdown(f"""
    <div style="font-size:13px;font-weight:600;color:#202124;margin:20px 0 10px;
         padding-bottom:6px;border-bottom:2px solid #1a73e8;display:inline-block">{text}</div>
    """, unsafe_allow_html=True)


def chart_container(title):
    """Render chart title. Use with st.plotly_chart after this call."""
    st.markdown(f'<div style="font-size:12px;font-weight:500;color:#202124;margin-bottom:8px">{title}</div>',
                unsafe_allow_html=True)


def format_inr(value, cr_threshold=10_000_000):
    """Format Indian Rupees with Cr/L suffix."""
    if abs(value) >= cr_threshold:
        return f"₹{value/10_000_000:.2f} Cr"
    elif abs(value) >= 100_000:
        return f"₹{value/100_000:.1f}L"
    else:
        return f"₹{value:,.0f}"


def callout_box(text, style='info'):
    """Render an info/warning callout."""
    if style == 'warning':
        border = '#e8710a'
        bg = '#fef7e0'
        color = '#b06000'
    elif style == 'critical':
        border = '#d93025'
        bg = '#fce8e6'
        color = '#a50e0e'
    else:
        border = '#1a73e8'
        bg = '#e8f0fe'
        color = '#174ea6'
    st.markdown(f"""
    <div style="background:{bg};border-left:3px solid {border};border-radius:0 4px 4px 0;
         padding:10px 14px;font-size:12px;color:{color};line-height:1.5;margin:6px 0">{text}</div>
    """, unsafe_allow_html=True)


def styled_plotly(fig, height=280, **kwargs):
    """Apply standard layout to a plotly figure and render it."""
    layout = {**CHART_LAYOUT, 'height': height, **kwargs}
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


GLOBAL_CSS = """
<style>
/* Sidebar */
section[data-testid="stSidebar"]{background:#1a1a2e;}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown div{color:#e0e0e0}
section[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,0.08)}

/* Style sidebar radio buttons as modern nav links */
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label > div:first-child { display: none !important; }
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    padding: 10px 14px !important;
    border-radius: 6px !important;
    margin-bottom: 4px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(255, 255, 255, 0.05) !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"] {
    background: rgba(26, 115, 232, 0.15) !important;
    border-left: 3px solid #8ab4f8 !important;
    border-radius: 0 6px 6px 0 !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {
    font-size: 15px !important;
    color: #e0e0e0 !important;
    margin: 0 !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"] p {
    color: #8ab4f8 !important;
    font-weight: 600 !important;
}

/* General cleanup — hide Streamlit chrome */
#MainMenu, footer { display: none !important; }
header[data-testid="stHeader"] { display: none !important; }
/* Keep the sidebar toggle button visible when sidebar is collapsed */
div[data-testid="stSidebarCollapsedControl"] { display: flex !important; }
.block-container { padding: 24px 24px 40px !important; max-width: 100% !important; }
.stApp{background:#f8f9fa}

/* Form elements */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label{font-size:12px!important;color:#5f6368!important;font-weight:500!important}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{gap:0;border-bottom:1px solid #dadce0}
.stTabs [data-baseweb="tab"]{font-size:13px;padding:8px 16px;color:#5f6368}
.stTabs [aria-selected="true"]{color:#1a73e8!important;border-bottom:2px solid #1a73e8!important}

/* Dataframe */
div[data-testid="stDataFrame"]{border:1px solid #dadce0;border-radius:4px}
</style>
"""
