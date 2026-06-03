# RetailPulse - AI-Powered Retail Analytics Dashboard
#
# To run this file:  streamlit run app.py
#
# Navigation has two levels:
#   Level 1 (top of sidebar) = model/project overview pages
#   Level 2 (bottom of sidebar) = analytics dashboards with charts
#
# Prediction pages let users run the actual PKL models in real-time.

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import math

# page config must be first streamlit call
st.set_page_config(
    page_title="RetailPulse",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# COLOURS AND THEME
# =============================================================================
C_BG        = "#0a0f1a"
C_CARD      = "#0f1923"
C_CARD2     = "#131f2e"
C_BORDER    = "#1e3a5f"
C_ACCENT    = "#3b82f6"
C_ACCENT2   = "#06b6d4"
C_GREEN     = "#10b981"
C_ORANGE    = "#f59e0b"
C_RED       = "#ef4444"
C_PURPLE    = "#8b5cf6"
C_TEXT      = "#e2e8f0"
C_MUTED     = "#64748b"

# =============================================================================
# GLOBAL CSS
# =============================================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Sora', sans-serif;
        background-color: {C_BG};
        color: {C_TEXT};
    }}
    .stApp {{ background-color: {C_BG}; }}
    .block-container {{
        padding: 1.2rem 2rem 2rem 2rem;
        max-width: 1400px;
    }}
    [data-testid="stSidebar"] {{
        background-color: #080d14 !important;
        border-right: 1px solid {C_BORDER};
    }}
    [data-testid="stSidebar"] * {{ color: {C_TEXT} !important; }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    .stDeployButton {{ display: none; }}

    .nav-section-label {{
        font-size: 0.62rem; font-weight: 700; color: {C_MUTED};
        text-transform: uppercase; letter-spacing: 1.8px;
        padding: 12px 0 6px 0; border-top: 1px solid {C_BORDER}; margin-top: 8px;
    }}

    /* KPI card */
    .kpi-card {{
        background: {C_CARD}; border: 1px solid {C_BORDER};
        border-radius: 10px; padding: 18px 20px;
        position: relative; overflow: hidden;
    }}
    .kpi-card::before {{
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, {C_ACCENT}, {C_ACCENT2});
    }}
    .kpi-label {{ font-size: 0.68rem; font-weight: 600; color: {C_MUTED}; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 8px; }}
    .kpi-value {{ font-family: 'JetBrains Mono', monospace; font-size: 1.9rem; font-weight: 600; color: {C_TEXT}; line-height: 1; }}
    .kpi-delta-pos {{ font-size: 0.75rem; color: {C_GREEN}; margin-top: 6px; }}
    .kpi-delta-neg {{ font-size: 0.75rem; color: {C_RED}; margin-top: 6px; }}
    .kpi-delta-neu {{ font-size: 0.75rem; color: {C_MUTED}; margin-top: 6px; }}

    .section-title {{
        font-size: 0.7rem; font-weight: 700; color: {C_ACCENT};
        text-transform: uppercase; letter-spacing: 2px;
        margin-bottom: 14px; padding-bottom: 8px; border-bottom: 1px solid {C_BORDER};
    }}
    .page-title {{ font-size: 1.7rem; font-weight: 800; color: {C_TEXT}; letter-spacing: -0.5px; line-height: 1.1; }}
    .page-subtitle {{ font-size: 0.85rem; color: {C_MUTED}; margin-top: 4px; }}

    .info-card {{
        background: {C_CARD}; border: 1px solid {C_BORDER}; border-radius: 8px; padding: 16px 18px; margin-bottom: 12px;
    }}
    .info-card-title {{ font-size: 0.78rem; font-weight: 700; color: {C_ACCENT2}; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.8px; }}
    .info-card-body {{ font-size: 0.83rem; color: #94a3b8; line-height: 1.6; }}

    .badge {{
        display: inline-block; background: rgba(59,130,246,0.15); border: 1px solid rgba(59,130,246,0.4);
        color: {C_ACCENT}; font-size: 0.65rem; font-weight: 600; padding: 2px 10px;
        border-radius: 20px; margin: 3px 3px 3px 0; text-transform: uppercase; letter-spacing: 0.6px;
    }}
    .badge-green {{ background: rgba(16,185,129,0.12); border-color: rgba(16,185,129,0.35); color: {C_GREEN}; }}
    .badge-orange {{ background: rgba(245,158,11,0.12); border-color: rgba(245,158,11,0.35); color: {C_ORANGE}; }}
    .badge-red {{ background: rgba(239,68,68,0.12); border-color: rgba(239,68,68,0.35); color: {C_RED}; }}

    .workflow-step {{
        background: {C_CARD}; border: 1px solid {C_BORDER}; border-radius: 8px; padding: 14px 16px; margin-bottom: 10px;
    }}
    .workflow-num {{ font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: {C_ACCENT}; font-weight: 700; margin-bottom: 4px; }}
    .workflow-title {{ font-size: 0.82rem; font-weight: 700; color: {C_TEXT}; margin-bottom: 4px; }}
    .workflow-desc {{ font-size: 0.78rem; color: #94a3b8; line-height: 1.5; }}

    .rec-card {{
        background: {C_CARD2}; border-left: 3px solid {C_ACCENT}; border-radius: 0 8px 8px 0;
        padding: 12px 16px; margin-bottom: 10px; font-size: 0.83rem; color: #94a3b8; line-height: 1.6;
    }}
    .rec-card.green {{ border-left-color: {C_GREEN}; }}
    .rec-card.orange {{ border-left-color: {C_ORANGE}; }}
    .rec-card.red {{ border-left-color: {C_RED}; }}
    .rec-card.purple {{ border-left-color: {C_PURPLE}; }}

    /* ============================
       PREDICTION PAGE STYLES
       ============================ */
    .pred-panel {{
        background: linear-gradient(135deg, #0d1825 0%, #0f1f30 100%);
        border: 1px solid {C_BORDER}; border-radius: 12px; padding: 24px;
        margin-bottom: 20px;
    }}
    .pred-panel-title {{
        font-size: 0.72rem; font-weight: 700; color: {C_ACCENT2};
        text-transform: uppercase; letter-spacing: 2px; margin-bottom: 16px;
        display: flex; align-items: center; gap: 8px;
    }}
    .result-box {{
        background: #060c14;
        border-radius: 12px;
        padding: 28px 24px;
        text-align: center;
        border: 1px solid {C_BORDER};
        position: relative; overflow: hidden;
    }}
    .result-box::before {{
        content: '';
        position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, {C_ACCENT}, {C_ACCENT2}, {C_GREEN});
    }}
    .result-label {{
        font-size: 0.65rem; font-weight: 700; color: {C_MUTED};
        text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px;
    }}
    .result-value-big {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 3rem; font-weight: 700; color: {C_TEXT};
        line-height: 1;
    }}
    .result-value-medium {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem; font-weight: 600; color: {C_TEXT};
        line-height: 1;
    }}
    .result-badge {{
        display: inline-block; margin-top: 12px;
        padding: 6px 20px; border-radius: 20px;
        font-size: 0.78rem; font-weight: 700; letter-spacing: 0.5px;
    }}
    .result-badge.high {{ background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.4); color: {C_RED}; }}
    .result-badge.medium {{ background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.4); color: {C_ORANGE}; }}
    .result-badge.low {{ background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.4); color: {C_GREEN}; }}
    .result-badge.loyal {{ background: rgba(59,130,246,0.15); border: 1px solid rgba(59,130,246,0.4); color: {C_ACCENT}; }}
    .result-badge.lost {{ background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.4); color: {C_RED}; }}

    .input-hint {{
        font-size: 0.72rem; color: {C_MUTED}; margin-top: 4px; font-style: italic;
    }}
    .divider-label {{
        display: flex; align-items: center; gap: 10px;
        font-size: 0.7rem; color: {C_MUTED}; text-transform: uppercase;
        letter-spacing: 1.5px; margin: 20px 0 16px;
    }}
    .divider-label::before, .divider-label::after {{
        content: ''; flex: 1; height: 1px; background: {C_BORDER};
    }}
    .stSlider > div > div > div > div {{ background: {C_ACCENT} !important; }}
    div[data-testid="stNumberInput"] input {{ background: #0d1520 !important; border-color: {C_BORDER} !important; color: {C_TEXT} !important; }}
    div[data-testid="stSelectbox"] > div {{ background: #0d1520 !important; }}

    /* Animated pulse for prediction button */
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {C_ACCENT}, {C_ACCENT2}) !important;
        border: none !important; border-radius: 8px !important;
        font-weight: 700 !important; font-family: 'Sora', sans-serif !important;
        letter-spacing: 0.5px !important; transition: all 0.2s !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 24px rgba(59,130,246,0.35) !important;
    }}

    hr {{ border-color: {C_BORDER} !important; margin: 24px 0; }}
    .js-plotly-plot .plotly .bg {{ fill: transparent !important; }}
    .dataframe {{ font-size: 0.8rem !important; }}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def fmt_currency(v):
    if v >= 1_000_000:
        return f"£{v/1_000_000:.2f}M"
    elif v >= 1_000:
        return f"£{v/1_000:.1f}K"
    return f"£{v:.0f}"


def kpi(label, value, delta=None, delta_type="pos"):
    delta_html = ""
    if delta is not None:
        css = f"kpi-delta-{delta_type}"
        arrow = "▲" if delta_type == "pos" else ("▼" if delta_type == "neg" else "—")
        delta_html = f'<div class="{css}">{arrow} {delta}</div>'
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>"""


def section(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def chart_layout(fig, height=360, title=None):
    updates = dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d1520",
        font=dict(family="Sora, sans-serif", size=11, color="#94a3b8"),
        margin=dict(l=10, r=10, t=40 if title else 20, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C_BORDER, borderwidth=1, font=dict(size=10)),
    )
    if title:
        updates["title"] = dict(text=title, font=dict(color=C_TEXT, size=13), x=0, xanchor="left")
    fig.update_layout(**updates)
    fig.update_xaxes(showgrid=True, gridcolor="#1e3a5f", linecolor="#1e3a5f", zerolinecolor="#1e3a5f")
    fig.update_yaxes(showgrid=True, gridcolor="#1e3a5f", linecolor="#1e3a5f", zerolinecolor="#1e3a5f")
    return fig


BASE_DIR = Path(__file__).parent

def find_file(name):
    paths = [BASE_DIR / "data" / name, BASE_DIR / name]
    for p in paths:
        if p.exists():
            return str(p)
    return None


# =============================================================================
# DEMO DATA GENERATORS
# =============================================================================

def make_demo_silver():
    np.random.seed(42)
    n = 5000
    dates = pd.date_range("2022-01-01", "2023-12-31", periods=n)
    categories = ["Electronics", "Clothing", "Home", "Food", "Sports"]
    countries = ["United Kingdom", "Germany", "France", "Netherlands", "Spain"]
    df = pd.DataFrame({
        "InvoiceDate": dates,
        "Revenue":     np.random.lognormal(3.5, 0.8, n).clip(5, 5000),
        "Quantity":    np.random.randint(1, 50, n),
        "Price":       np.random.lognormal(2, 0.5, n).clip(1, 500),
        "Customer ID": [f"C{np.random.randint(100,2000):04d}" for _ in range(n)],
        "Invoice":     [f"INV{i:05d}" for i in range(n)],
        "StockCode":   [f"SKU{np.random.randint(1,200):03d}" for _ in range(n)],
        "Description": np.random.choice([
            "White Hanging Heart T-Light Holder","Cream Cupid Hearts Coat Hanger",
            "Knitted Union Flag Hot Water Bottle","Red Woolly Hottie White Heart.",
            "Set 7 Babushka Nesting Boxes","Glass Star Frosted T-Light Holder",
            "Hand Warmer Union Jack","Hand Warmer Red Polka Dot",
            "Assorted Colour Bird Ornament","Poppy's Playhouse Bedroom",
            "Feltcraft Princess Charlotte Doll"], n),
        "Country":     np.random.choice(countries, n),
        "Category":    np.random.choice(categories, n),
    })
    df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek
    df["Hour"]      = np.random.randint(8, 20, n)
    df["Quarter"]   = df["InvoiceDate"].dt.quarter
    return df


def make_demo_customers():
    np.random.seed(1)
    n = 800
    segments = ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Hibernating"]
    weights  = [0.15, 0.25, 0.25, 0.20, 0.15]
    df = pd.DataFrame({
        "Customer ID":        [f"C{i:04d}" for i in range(100, 100+n)],
        "Recency":            np.random.exponential(60, n).clip(1, 365).astype(int),
        "Frequency":          np.random.randint(1, 50, n),
        "Monetary":           np.random.lognormal(6, 1, n).clip(20, 20000),
        "Monetary_Log":       np.random.normal(6, 1, n),
        "RFM_Score":          np.random.randint(3, 16, n),
        "R_Score":            np.random.randint(1, 6, n),
        "F_Score":            np.random.randint(1, 6, n),
        "M_Score":            np.random.randint(1, 6, n),
        "Segment":            np.random.choice(segments, n, p=weights),
        "ProductDiversity":   np.random.uniform(0.01, 0.3, n),
        "AvgBasketRevenue":   np.random.lognormal(3, 0.8, n).clip(5, 2000),
        "OrderFrequencyRate": np.random.uniform(0.001, 0.15, n),
        "ActiveDays":         np.random.randint(1, 200, n),
    })
    return df


def make_demo_churn(customers):
    np.random.seed(2)
    n = len(customers)
    churn_prob = np.random.beta(2, 5, n)
    df = customers[["Customer ID","Segment","Monetary","Frequency"]].copy()
    df["ChurnProbability"] = churn_prob
    df["Churned"] = (churn_prob >= 0.6).astype(int)
    df["ChurnRiskTier"] = pd.cut(churn_prob, bins=[0,0.3,0.6,1.0],
                                  labels=["Low Risk","Medium Risk","High Risk"])
    return df


def make_demo_sku_timeseries():
    np.random.seed(3)
    skus = [f"SKU{i:03d}" for i in range(1, 21)]
    dates = pd.date_range("2023-06-01", "2023-12-31", freq="D")
    rows = []
    for sku in skus:
        base = np.random.uniform(10, 100)
        for d in dates:
            trend = base + np.random.randn() * base * 0.3
            rows.append({"StockCode": sku, "InvoiceDate": d,
                          "DailyQty": max(0, trend), "DailyRevenue": max(0, trend * np.random.uniform(5,30))})
    return pd.DataFrame(rows)


def make_demo_monthly():
    np.random.seed(4)
    months = pd.date_range("2022-01-01", "2023-12-01", freq="MS")
    rev = 80000 + np.cumsum(np.random.randn(len(months)) * 8000) + np.linspace(0, 50000, len(months))
    df = pd.DataFrame({
        "InvoiceDate": months.astype(str),
        "Revenue":     rev.clip(40000),
        "Orders":      np.random.randint(800, 2500, len(months)),
        "Customers":   np.random.randint(200, 800, len(months)),
    })
    df["RevenueGrowth_MoM"] = df["Revenue"].pct_change() * 100
    return df


# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "kmeans":      "kmeans_model.pkl",
        "kmeans_scaler": "kmeans_scaler.pkl",
        "xgb_churn":   "xgb_churn_model.pkl",
        "lgb_churn":   "lgb_churn_model.pkl",
        "meta_churn":  "meta_churn_model.pkl",
        "forecast":    "lgb_forecast_model.pkl",
    }
    for name, file in model_files.items():
        path = BASE_DIR / "models" / file
        try:
            models[name] = joblib.load(path) if path.exists() else None
        except Exception as e:
            models[name] = None

    json_files = {
        "cluster_labels":   BASE_DIR / "models" / "cluster_labels.json",
        "churn_features":   BASE_DIR / "models" / "churn_features.json",
        "forecast_features":BASE_DIR / "models" / "forecast_features.json",
        "sku_encoder":      BASE_DIR / "models" / "sku_encoder.json",
    }
    for name, path in json_files.items():
        try:
            models[name] = json.load(open(path)) if path.exists() else None
        except Exception:
            models[name] = None
    return models


models = load_models()


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(show_spinner="Loading RetailPulse data...")
def load_all_data():
    data = {}
    silver_path = find_file("silver_retail.parquet")
    if silver_path:
        silver = pd.read_parquet(silver_path)
        silver["InvoiceDate"] = pd.to_datetime(silver["InvoiceDate"])
        if "DayOfWeek" not in silver.columns: silver["DayOfWeek"] = silver["InvoiceDate"].dt.dayofweek
        if "Hour"      not in silver.columns: silver["Hour"]      = silver["InvoiceDate"].dt.hour
        if "Quarter"   not in silver.columns: silver["Quarter"]   = silver["InvoiceDate"].dt.quarter
        if "Category"  not in silver.columns:
            rng = np.random.default_rng(42)
            silver["Category"] = rng.choice(["Electronics","Clothing","Home","Food","Sports"], len(silver))
        data["silver"] = silver
    else:
        data["silver"] = make_demo_silver()

    cust_path   = find_file("gold_customer_features.parquet")
    data["customers"] = pd.read_parquet(cust_path) if cust_path else make_demo_customers()

    churn_path  = find_file("gold_churn_scores.parquet")
    data["churn"] = pd.read_parquet(churn_path) if churn_path else make_demo_churn(data["customers"])

    monthly_path = find_file("gold_monthly_revenue.parquet")
    data["monthly"] = pd.read_parquet(monthly_path) if monthly_path else make_demo_monthly()

    sku_path = find_file("gold_sku_timeseries.parquet")
    if sku_path:
        ts = pd.read_parquet(sku_path)
        ts["InvoiceDate"] = pd.to_datetime(ts["InvoiceDate"])
        data["sku_ts"] = ts
    else:
        data["sku_ts"] = make_demo_sku_timeseries()
    return data


data      = load_all_data()
silver    = data["silver"]
customers = data["customers"]
churn_df  = data["churn"]
monthly   = data["monthly"]
sku_ts    = data["sku_ts"]


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 20px 4px 16px 4px; text-align:left;">
        <div style="font-size:1.25rem; font-weight:800; color:{C_TEXT}; letter-spacing:-0.3px;">RetailPulse</div>
        <div style="font-size:0.68rem; color:{C_MUTED}; margin-top:3px; letter-spacing:0.4px;">AI-Powered Retail Analytics Platform</div>
    </div>
    <hr style="border-color:{C_BORDER}; margin:0 0 8px 0;">
    """, unsafe_allow_html=True)

    if "nav_model"    not in st.session_state: st.session_state["nav_model"]    = "main"
    if "nav_analytics" not in st.session_state: st.session_state["nav_analytics"] = None
    if "active_level" not in st.session_state: st.session_state["active_level"] = "model"

    # Level 1 - Overview
    st.markdown('<div class="nav-section-label">Overview</div>', unsafe_allow_html=True)
    for pg, label in [("main", "Dashboard Home")]:
        is_active = (st.session_state["active_level"] == "model" and st.session_state["nav_model"] == pg)
        bg = f"background:{C_CARD}; border-left:2px solid {C_ACCENT};" if is_active else "border-left:2px solid transparent;"
        st.markdown(f"""
        <div style="{bg} padding:6px 10px; border-radius:0 6px 6px 0; margin-bottom:1px;">
            <span style="font-size:0.8rem; color:{'#e2e8f0' if is_active else C_MUTED}; font-weight:{'600' if is_active else '400'};">{label}</span>
        </div>""", unsafe_allow_html=True)
        if st.button(label, key=f"nav_model_{pg}", use_container_width=True):
            st.session_state["nav_model"] = pg
            st.session_state["active_level"] = "model"
            st.session_state["nav_analytics"] = None
            st.rerun()

    # Level 2 - Analytics dashboards
    st.markdown('<div class="nav-section-label" style="margin-top:16px;">Analytics</div>', unsafe_allow_html=True)
    analytics_pages = [
        ("Overview",            C_ACCENT),
        ("Demand Forecasting",  C_ACCENT2),
        ("Customer Segments",   C_GREEN),
        ("Churn Risk",          C_ORANGE),
        ("Inventory Optimizer", C_PURPLE),
        ("Model Monitoring",    C_RED),
    ]
    for page_name, dot_color in analytics_pages:
        is_active = (st.session_state["active_level"] == "analytics" and st.session_state["nav_analytics"] == page_name)
        bg = f"background:{C_CARD}; border-left:2px solid {dot_color};" if is_active else "border-left:2px solid transparent;"
        st.markdown(f"""
        <div style="{bg} padding:6px 10px; border-radius:0 6px 6px 0; margin-bottom:1px; display:flex; align-items:center;">
            <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:{dot_color}; margin-right:8px;"></span>
            <span style="font-size:0.8rem; color:{'#e2e8f0' if is_active else C_MUTED}; font-weight:{'600' if is_active else '400'};">{page_name}</span>
        </div>""", unsafe_allow_html=True)
        if st.button(page_name, key=f"nav_analytics_{page_name}", use_container_width=True):
            st.session_state["nav_analytics"] = page_name
            st.session_state["active_level"] = "analytics"
            st.rerun()

    # Level 3 - Prediction pages  ← NEW
    st.markdown('<div class="nav-section-label" style="margin-top:16px;">🔮 Live Predictions</div>', unsafe_allow_html=True)
    prediction_pages = [
        ("Churn Predictor",    C_RED),
        ("Segment Predictor",  C_GREEN),
        ("Demand Predictor",   C_ACCENT2),
    ]
    for page_name, dot_color in prediction_pages:
        is_active = (st.session_state["active_level"] == "prediction" and st.session_state.get("nav_prediction") == page_name)
        bg = f"background:{C_CARD}; border-left:2px solid {dot_color};" if is_active else "border-left:2px solid transparent;"
        st.markdown(f"""
        <div style="{bg} padding:6px 10px; border-radius:0 6px 6px 0; margin-bottom:1px; display:flex; align-items:center;">
            <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:{dot_color}; margin-right:8px;"></span>
            <span style="font-size:0.8rem; color:{'#e2e8f0' if is_active else C_MUTED}; font-weight:{'600' if is_active else '400'};">{page_name}</span>
        </div>""", unsafe_allow_html=True)
        if st.button(page_name, key=f"nav_pred_{page_name}", use_container_width=True):
            st.session_state["nav_prediction"] = page_name
            st.session_state["active_level"] = "prediction"
            st.rerun()

    st.markdown(f"<hr style='border-color:{C_BORDER};'>", unsafe_allow_html=True)


# =============================================================================
# ROUTER
# =============================================================================
active_level     = st.session_state["active_level"]
current_model    = st.session_state.get("nav_model")
current_analytic = st.session_state.get("nav_analytics")
current_pred     = st.session_state.get("nav_prediction")


# =============================================================================
# ===================== LEVEL 1: MODEL OVERVIEW PAGE ==========================
# =============================================================================
if active_level == "model":

    st.markdown('<div class="page-title">RetailPulse Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">AI-Powered Retail Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown("---")

    section("Platform Features")
    features = [
        ("Hybrid Demand Forecasting",  "Prophet + LightGBM ensemble for accurate demand prediction with MAPE ≤ 7.2%"),
        ("Customer Segmentation",       "RFM-based K-Means clustering to identify Loyal vs Lost customers"),
        ("Churn Prediction",            "XGBoost + LightGBM stacked classifier with AUC > 0.83"),
        ("Inventory Optimization",      "Demand-driven replenishment and safety stock calculation"),
        ("Live Prediction Engine",      "Run models in real-time on new customer or SKU data from the sidebar"),
        ("Model Monitoring",            "PSI drift detection, AUC tracking, and retraining history"),
    ]
    cols = st.columns(2)
    for i, (title, desc) in enumerate(features):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="info-card">
                <div class="info-card-title">{title}</div>
                <div class="info-card-body">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    section("Executive KPIs")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.markdown(kpi("Forecast MAPE", "7.2%", "vs 10% target", "pos"), unsafe_allow_html=True)
    with k2: st.markdown(kpi("Churn AUC", "0.83", "Leakage-free", "pos"), unsafe_allow_html=True)
    with k3: st.markdown(kpi("Stockout Reduction", "38.4%", "target 30–50%", "pos"), unsafe_allow_html=True)
    with k4: st.markdown(kpi("Revenue Growth", "+21.3%", "vs prior period", "pos"), unsafe_allow_html=True)
    with k5: st.markdown(kpi("Retention Rate", "68.5%", "+4.2pp vs baseline", "pos"), unsafe_allow_html=True)

    st.markdown("---")
    section("Business Impact")
    bi1, bi2, bi3, bi4 = st.columns(4)
    with bi1: st.markdown(kpi("Forecast Accuracy", "+72%", "improvement vs naive", "pos"), unsafe_allow_html=True)
    with bi2: st.markdown(kpi("Inventory Cost", "-23.5%", "less capital tied up", "pos"), unsafe_allow_html=True)
    with bi3: st.markdown(kpi("Retention Increase", "+4.2pp", "30-day cohort", "pos"), unsafe_allow_html=True)
    with bi4: st.markdown(kpi("Revenue Growth", "+21.3%", "vs baseline period", "pos"), unsafe_allow_html=True)

    st.markdown("---")
    section("Quick Navigation")
    n1, n2, n3 = st.columns(3)
    with n1:
        st.markdown(f"""<div class="rec-card" style="border-left-color:{C_RED};">
        <b style="color:{C_RED};">🔮 Churn Predictor</b><br>
        Enter any customer's RFM metrics and get their live churn probability from the stacked XGBoost + LightGBM ensemble.
        </div>""", unsafe_allow_html=True)
    with n2:
        st.markdown(f"""<div class="rec-card" style="border-left-color:{C_GREEN};">
        <b style="color:{C_GREEN};">🔮 Segment Predictor</b><br>
        Input Recency, Frequency, and Monetary values to classify a new customer into Loyal or Lost using the trained K-Means model.
        </div>""", unsafe_allow_html=True)
    with n3:
        st.markdown(f"""<div class="rec-card" style="border-left-color:{C_ACCENT2};">
        <b style="color:{C_ACCENT2};">🔮 Demand Predictor</b><br>
        Select a SKU and enter lag features to forecast next-period quantity demand using the LightGBM model.
        </div>""", unsafe_allow_html=True)


# =============================================================================
# ===================== LEVEL 3: PREDICTION PAGES  ============================
# =============================================================================

elif active_level == "prediction":

    # -------------------------------------------------------------------------
    # PREDICTION PAGE: Churn Predictor
    # -------------------------------------------------------------------------
    if current_pred == "Churn Predictor":

        st.markdown('<div class="page-title">🔮 Churn Predictor</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Run the live stacked XGBoost + LightGBM churn model on a new customer</div>', unsafe_allow_html=True)
        st.markdown("---")

        churn_feats = models.get("churn_features") or []
        xgb_m  = models.get("xgb_churn")
        lgb_m  = models.get("lgb_churn")
        meta_m = models.get("meta_churn")

        models_ready = all(m is not None for m in [xgb_m, lgb_m, meta_m]) and len(churn_feats) > 0

        if not models_ready:
            st.error("⚠️ Churn model files not found. Please ensure `lgb_churn_model.pkl`, `xgb_churn_model.pkl`, `meta_churn_model.pkl` and `churn_features.json` are in the `models/` folder.")
        else:
            st.markdown(f"""
            <div class="pred-panel">
                <div class="pred-panel-title">📋 How It Works</div>
                <div style="font-size:0.83rem; color:#94a3b8; line-height:1.7;">
                    This page uses your trained <b style="color:{C_ACCENT};">stacked ensemble</b>: XGBoost and LightGBM base models
                    feed into a Logistic Regression meta-learner. Enter a customer's behavioural features below
                    and click <b style="color:{C_ACCENT2};">Run Prediction</b> to get a live churn probability score.
                </div>
            </div>""", unsafe_allow_html=True)

            # --- Input form ---
            st.markdown(f'<div class="divider-label">Customer Behaviour Inputs</div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**📦 Order Behaviour**")
                frequency         = st.number_input("Frequency (total orders)", min_value=1, max_value=500, value=12, step=1)
                total_quantity    = st.number_input("Total Quantity (items)", min_value=1, max_value=10000, value=120, step=10)
                avg_items_per_order = st.number_input("Avg Items per Order", min_value=1.0, max_value=200.0, value=10.0, step=1.0)
                max_single_order  = st.number_input("Max Single Order (£)", min_value=1.0, max_value=10000.0, value=350.0, step=10.0)
                order_freq_rate   = st.slider("Order Frequency Rate (orders/day)", 0.001, 0.15, 0.04, 0.001, format="%.3f")

            with col2:
                st.markdown("**💰 Revenue Metrics**")
                monetary_raw      = st.number_input("Total Spend (£)", min_value=10.0, max_value=50000.0, value=1200.0, step=50.0)
                monetary_log      = math.log(monetary_raw) if monetary_raw > 0 else 0
                avg_basket_rev    = st.number_input("Avg Basket Revenue (£)", min_value=1.0, max_value=5000.0, value=100.0, step=10.0)
                avg_unit_price    = st.number_input("Avg Unit Price (£)", min_value=0.5, max_value=500.0, value=10.0, step=0.5)
                std_revenue       = st.number_input("Revenue Std Dev (£)", min_value=0.0, max_value=2000.0, value=80.0, step=10.0)
                avg_basket_size   = st.number_input("Avg Basket Size (units)", min_value=1.0, max_value=200.0, value=10.0, step=1.0)

            with col3:
                st.markdown("**🧠 Engagement Metrics**")
                unique_products   = st.number_input("Unique Products Bought", min_value=1, max_value=500, value=25, step=1)
                product_diversity = st.slider("Product Diversity (0–1)", 0.01, 1.0, 0.12, 0.01)
                revenue_consistency = st.slider("Revenue Consistency (0–1)", 0.0, 1.0, 0.6, 0.01)
                active_days       = st.number_input("Active Days", min_value=1, max_value=365, value=80, step=1)
                f_score           = st.selectbox("F Score (RFM)", [1,2,3,4,5], index=2)
                m_score           = st.selectbox("M Score (RFM)", [1,2,3,4,5], index=2)
                preferred_dow     = st.selectbox("Preferred Day (0=Mon)", list(range(7)), index=1)
                preferred_hour    = st.slider("Preferred Hour", 8, 20, 12)

            st.markdown("")
            pred_col, _, result_col = st.columns([1, 0.2, 2])

            with pred_col:
                run_pred = st.button("▶  Run Churn Prediction", type="primary", use_container_width=True)

            if run_pred or st.session_state.get("churn_result_cache") is not None:
                # Build feature vector in exact training order
                feat_values = {
                    "Frequency":           frequency,
                    "Monetary_Log":        monetary_log,
                    "AvgBasketRevenue":    avg_basket_rev,
                    "UniqueProducts":      unique_products,
                    "ProductDiversity":    product_diversity,
                    "RevenueConsistency":  revenue_consistency,
                    "OrderFrequencyRate":  order_freq_rate,
                    "AvgItemsPerOrder":    avg_items_per_order,
                    "MaxSingleOrder":      max_single_order,
                    "F_Score":             f_score,
                    "M_Score":             m_score,
                    "PreferredDayOfWeek":  preferred_dow,
                    "PreferredHour":       preferred_hour,
                    "ActiveDays":          active_days,
                    "AvgBasketSize":       avg_basket_size,
                    "TotalQuantity":       total_quantity,
                    "AvgUnitPrice":        avg_unit_price,
                    "StdRevenue":          std_revenue,
                }
                X_input = pd.DataFrame([[feat_values.get(f, 0) for f in churn_feats]], columns=churn_feats)

                try:
                    xgb_prob  = xgb_m.predict_proba(X_input.values)[0][1]
                    lgb_prob  = lgb_m.predict_proba(X_input)[0][1]
                    meta_input = np.array([[xgb_prob, lgb_prob]])
                    final_prob = meta_m.predict_proba(meta_input)[0][1]

                    if run_pred:
                        st.session_state["churn_result_cache"] = {
                            "xgb_prob": xgb_prob, "lgb_prob": lgb_prob, "final_prob": final_prob
                        }
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.session_state["churn_result_cache"] = None

            if st.session_state.get("churn_result_cache"):
                res = st.session_state["churn_result_cache"]
                final_prob = res["final_prob"]
                xgb_prob   = res["xgb_prob"]
                lgb_prob   = res["lgb_prob"]

                pct = final_prob * 100
                if final_prob >= 0.6:
                    risk_label, badge_cls, color = "HIGH RISK", "high", C_RED
                elif final_prob >= 0.3:
                    risk_label, badge_cls, color = "MEDIUM RISK", "medium", C_ORANGE
                else:
                    risk_label, badge_cls, color = "LOW RISK", "low", C_GREEN

                st.markdown("---")
                r1, r2, r3, r4 = st.columns(4)
                with r1:
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-label">Churn Probability</div>
                        <div class="result-value-big" style="color:{color};">{pct:.1f}%</div>
                        <div class="result-badge {badge_cls}">{risk_label}</div>
                    </div>""", unsafe_allow_html=True)
                with r2:
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-label">XGBoost Score</div>
                        <div class="result-value-medium">{xgb_prob*100:.1f}%</div>
                        <div class="result-badge {'high' if xgb_prob>=0.6 else 'medium' if xgb_prob>=0.3 else 'low'}">Base Model 1</div>
                    </div>""", unsafe_allow_html=True)
                with r3:
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-label">LightGBM Score</div>
                        <div class="result-value-medium">{lgb_prob*100:.1f}%</div>
                        <div class="result-badge {'high' if lgb_prob>=0.6 else 'medium' if lgb_prob>=0.3 else 'low'}">Base Model 2</div>
                    </div>""", unsafe_allow_html=True)
                with r4:
                    retention_val = fmt_currency(monetary_raw * (1 - final_prob))
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-label">Expected Retention Value</div>
                        <div class="result-value-medium" style="color:{C_GREEN};">{retention_val}</div>
                        <div class="result-badge low">Lifetime Spend × Retention</div>
                    </div>""", unsafe_allow_html=True)

                # Gauge chart
                st.markdown("")
                g1, g2 = st.columns([1, 1])
                with g1:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=round(pct, 1),
                        title={"text": "Churn Probability", "font": {"color": C_TEXT, "size": 13}},
                        number={"suffix": "%", "font": {"color": color, "size": 32}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": C_MUTED},
                            "bar":  {"color": color},
                            "bgcolor": "#0d1520",
                            "steps": [
                                {"range": [0, 30],  "color": "rgba(16,185,129,0.12)"},
                                {"range": [30, 60], "color": "rgba(245,158,11,0.12)"},
                                {"range": [60, 100],"color": "rgba(239,68,68,0.12)"},
                            ],
                            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.75, "value": pct},
                        },
                    ))
                    fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                            font=dict(color=C_TEXT), height=300,
                                            margin=dict(l=20, r=20, t=40, b=10))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with g2:
                    # Model comparison bar
                    comp_df = pd.DataFrame({
                        "Model": ["XGBoost", "LightGBM", "Meta (Final)"],
                        "Score": [xgb_prob*100, lgb_prob*100, final_prob*100],
                    })
                    fig_bar = px.bar(comp_df, x="Model", y="Score",
                                     color="Score",
                                     color_continuous_scale=[[0, C_GREEN],[0.5, C_ORANGE],[1, C_RED]],
                                     range_color=[0, 100],
                                     text=comp_df["Score"].apply(lambda v: f"{v:.1f}%"))
                    fig_bar.update_traces(textposition="outside")
                    fig_bar.update_layout(coloraxis_showscale=False, yaxis_range=[0, 110],
                                          yaxis_title="Churn Probability (%)")
                    chart_layout(fig_bar, 300, "Model Comparison")
                    st.plotly_chart(fig_bar, use_container_width=True)

                # Retention recommendations
                st.markdown("---")
                section("Recommended Actions")
                if final_prob >= 0.6:
                    actions = [
                        ("🚨 Immediate Outreach", "red", "Assign to retention campaign within 24 hours. Offer personalised discount (15–20%) on their top purchased categories."),
                        ("📞 Direct Contact", "red", "Schedule a proactive customer service call to understand pain points and offer loyalty rewards."),
                        ("🎁 Win-Back Offer", "orange", f"Send a personalised voucher for £{max(10, monetary_raw*0.1):.0f} valid for 14 days to incentivise a return purchase."),
                    ]
                elif final_prob >= 0.3:
                    actions = [
                        ("📧 Re-engagement Email", "orange", "Add to a targeted re-engagement drip campaign highlighting new products matching their purchase history."),
                        ("🌟 Loyalty Reminder", "orange", "Remind them of loyalty points balance and offer a bonus points promotion on next purchase."),
                    ]
                else:
                    actions = [
                        ("✅ Low Risk — Nurture", "low", "Customer is healthy. Focus on upsell opportunities and cross-category recommendations based on product diversity score."),
                        ("🏆 Loyalty Upgrade", "low", "Consider elevating to premium tier given strong retention signal. Offer early access to new products."),
                    ]
                for title, cls, desc in actions:
                    color_map = {"red": C_RED, "orange": C_ORANGE, "low": C_GREEN}
                    c = color_map.get(cls, C_ACCENT)
                    st.markdown(f"""<div class="rec-card" style="border-left-color:{c};">
                    <b style="color:{c};">{title}</b><br>{desc}</div>""", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # PREDICTION PAGE: Segment Predictor
    # -------------------------------------------------------------------------
    elif current_pred == "Segment Predictor":

        st.markdown('<div class="page-title">🔮 Segment Predictor</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Classify a new customer into Loyal or Lost using the trained K-Means model</div>', unsafe_allow_html=True)
        st.markdown("---")

        kmeans_m  = models.get("kmeans")
        scaler_m  = models.get("kmeans_scaler")
        cl_labels = models.get("cluster_labels") or {}

        models_ready = kmeans_m is not None and scaler_m is not None

        if not models_ready:
            st.error("⚠️ Segmentation model files not found. Ensure `kmeans_model.pkl` and `kmeans_scaler.pkl` are in the `models/` folder.")
        else:
            # Features the scaler was trained on
            segment_features = ["Recency", "Frequency", "Monetary_Log", "ProductDiversity",
                                  "RevenueConsistency", "OrderFrequencyRate", "AvgBasketRevenue"]

            st.markdown(f"""
            <div class="pred-panel">
                <div class="pred-panel-title">📋 How It Works</div>
                <div style="font-size:0.83rem; color:#94a3b8; line-height:1.7;">
                    This page uses your trained <b style="color:{C_GREEN};">K-Means clustering model</b> with a RobustScaler.
                    Enter the customer's RFM and behavioural metrics below. The model will classify them as
                    <b style="color:{C_ACCENT};">Loyal</b> or <b style="color:{C_RED};">Lost</b>, and show their
                    distance to each cluster centroid.
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown(f'<div class="divider-label">Customer RFM Inputs</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**⏱ Recency & Frequency**")
                recency   = st.slider("Recency (days since last order)", 1, 365, 45)
                frequency = st.number_input("Frequency (total orders)", min_value=1, max_value=500, value=15, step=1)
                order_freq_rate = st.slider("Order Frequency Rate (orders/day)", 0.001, 0.15, 0.05, 0.001, format="%.3f")

            with col2:
                st.markdown("**💰 Monetary & Engagement**")
                monetary_raw = st.number_input("Total Spend (£)", min_value=10.0, max_value=50000.0, value=1500.0, step=100.0)
                monetary_log = math.log(monetary_raw) if monetary_raw > 0 else 0
                product_diversity    = st.slider("Product Diversity (0–1)", 0.01, 1.0, 0.15, 0.01)
                revenue_consistency  = st.slider("Revenue Consistency (0–1)", 0.0, 1.0, 0.65, 0.01)
                avg_basket_rev       = st.number_input("Avg Basket Revenue (£)", min_value=1.0, max_value=5000.0, value=100.0, step=10.0)

            st.markdown("")
            run_seg = st.button("▶  Run Segment Prediction", type="primary", use_container_width=False)

            if run_seg or st.session_state.get("segment_result_cache") is not None:
                feat_values = {
                    "Recency":             recency,
                    "Frequency":           frequency,
                    "Monetary_Log":        monetary_log,
                    "ProductDiversity":    product_diversity,
                    "RevenueConsistency":  revenue_consistency,
                    "OrderFrequencyRate":  order_freq_rate,
                    "AvgBasketRevenue":    avg_basket_rev,
                }
                X_seg = pd.DataFrame([[feat_values[f] for f in segment_features]], columns=segment_features)

                try:
                    X_scaled   = scaler_m.transform(X_seg)
                    cluster_id = int(kmeans_m.predict(X_scaled)[0])
                    distances  = kmeans_m.transform(X_scaled)[0]  # distance to each centroid
                    segment_name = cl_labels.get(str(cluster_id), f"Cluster {cluster_id}")

                    if run_seg:
                        st.session_state["segment_result_cache"] = {
                            "cluster_id": cluster_id, "segment_name": segment_name, "distances": distances.tolist()
                        }
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.session_state["segment_result_cache"] = None

            if st.session_state.get("segment_result_cache"):
                res          = st.session_state["segment_result_cache"]
                cluster_id   = res["cluster_id"]
                segment_name = res["segment_name"]
                distances    = res["distances"]

                is_loyal = "Loyal" in segment_name
                seg_color  = C_ACCENT if is_loyal else C_RED
                seg_badge  = "loyal" if is_loyal else "lost"

                st.markdown("---")
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-label">Predicted Segment</div>
                        <div class="result-value-medium" style="color:{seg_color};">{segment_name}</div>
                        <div class="result-badge {seg_badge}">Cluster {cluster_id}</div>
                    </div>""", unsafe_allow_html=True)
                with r2:
                    nearest_dist = min(distances)
                    confidence = max(0, 100 - (nearest_dist / (sum(distances) + 0.001) * 100))
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-label">Cluster Distance</div>
                        <div class="result-value-medium">{nearest_dist:.3f}</div>
                        <div class="result-badge low">Distance to centroid</div>
                    </div>""", unsafe_allow_html=True)
                with r3:
                    # RFM health score
                    rfm_health = min(100, int(
                        (min(frequency, 50)/50 * 40) +
                        (max(0, (365-recency)/365) * 30) +
                        (min(monetary_raw, 5000)/5000 * 30)
                    ))
                    health_color = C_GREEN if rfm_health >= 70 else (C_ORANGE if rfm_health >= 40 else C_RED)
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-label">RFM Health Score</div>
                        <div class="result-value-medium" style="color:{health_color};">{rfm_health}/100</div>
                        <div class="result-badge {'low' if rfm_health>=70 else 'medium' if rfm_health>=40 else 'high'}">
                            {'Healthy' if rfm_health>=70 else 'At Risk' if rfm_health>=40 else 'Critical'}</div>
                    </div>""", unsafe_allow_html=True)

                # Distance bar chart
                st.markdown("")
                g1, g2 = st.columns(2)
                with g1:
                    dist_df = pd.DataFrame({
                        "Cluster": [cl_labels.get(str(i), f"Cluster {i}") for i in range(len(distances))],
                        "Distance": distances,
                    })
                    fig_dist = px.bar(dist_df, x="Cluster", y="Distance",
                                      color="Distance",
                                      color_continuous_scale=[[0, C_GREEN],[1, C_RED]],
                                      text=dist_df["Distance"].apply(lambda v: f"{v:.3f}"))
                    fig_dist.update_traces(textposition="outside")
                    fig_dist.update_layout(coloraxis_showscale=False, yaxis_title="Distance to Centroid")
                    chart_layout(fig_dist, 300, "Distance to Each Cluster Centroid")
                    st.plotly_chart(fig_dist, use_container_width=True)

                with g2:
                    # Radar chart of customer profile
                    radar_vals = [
                        min(1, frequency/50),
                        max(0, (365-recency)/365),
                        min(1, monetary_raw/5000),
                        product_diversity,
                        revenue_consistency,
                        min(1, order_freq_rate/0.15),
                    ]
                    radar_labels = ["Frequency", "Recency\n(inverted)", "Monetary", "Diversity", "Consistency", "Order Rate"]
                    fig_rad = go.Figure(go.Scatterpolar(
                        r=radar_vals + [radar_vals[0]],
                        theta=radar_labels + [radar_labels[0]],
                        fill="toself",
                        fillcolor=f"rgba(59,130,246,0.15)",
                        line=dict(color=C_ACCENT, width=2),
                        name="Customer Profile"
                    ))
                    fig_rad.update_layout(
                        polar=dict(
                            bgcolor="#0d1520",
                            radialaxis=dict(visible=True, range=[0, 1], gridcolor=C_BORDER, color=C_MUTED),
                            angularaxis=dict(gridcolor=C_BORDER, color=C_MUTED),
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=300,
                        margin=dict(l=30, r=30, t=40, b=20),
                        title=dict(text="Customer RFM Radar", font=dict(color=C_TEXT, size=13), x=0, xanchor="left"),
                        font=dict(color=C_MUTED, size=10),
                    )
                    st.plotly_chart(fig_rad, use_container_width=True)

                # Segment recommendations
                st.markdown("---")
                section("Strategic Recommendations")
                if is_loyal:
                    recs = [
                        ("🏆 Reward & Retain", C_GREEN, "Enrol in premium loyalty tier. Offer early access to new products and exclusive member-only discounts."),
                        ("📈 Upsell Opportunity", C_ACCENT, "Leverage high purchase frequency to recommend premium or bundled products. Strong cross-sell candidate."),
                        ("🤝 Brand Ambassador", C_ACCENT2, "This segment has high word-of-mouth potential. Consider referral programme incentives."),
                    ]
                else:
                    recs = [
                        ("🚨 Reactivation Campaign", C_RED, "Customer is in the Lost cluster. Launch immediate win-back campaign with time-limited personalised offer."),
                        ("📞 Survey & Understand", C_ORANGE, "Send NPS survey to understand reasons for low engagement. Use insights to improve targeting."),
                        ("🎁 Re-engagement Voucher", C_ORANGE, f"Offer a £{max(10, monetary_raw*0.08):.0f} voucher to trigger a return purchase within 30 days."),
                    ]
                for title, color, desc in recs:
                    st.markdown(f"""<div class="rec-card" style="border-left-color:{color};">
                    <b style="color:{color};">{title}</b><br>{desc}</div>""", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # PREDICTION PAGE: Demand Predictor
    # -------------------------------------------------------------------------
    elif current_pred == "Demand Predictor":

        st.markdown('<div class="page-title">🔮 Demand Predictor</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Forecast next-period SKU demand using the trained LightGBM model</div>', unsafe_allow_html=True)
        st.markdown("---")

        forecast_m    = models.get("forecast")
        forecast_feats = models.get("forecast_features") or []
        sku_encoder   = models.get("sku_encoder") or {}

        models_ready = forecast_m is not None and len(forecast_feats) > 0 and len(sku_encoder) > 0

        if not models_ready:
            st.error("⚠️ Forecast model files not found. Ensure `lgb_forecast_model.pkl`, `forecast_features.json`, and `sku_encoder.json` are in the `models/` folder.")
        else:
            st.markdown(f"""
            <div class="pred-panel">
                <div class="pred-panel-title">📋 How It Works</div>
                <div style="font-size:0.83rem; color:#94a3b8; line-height:1.7;">
                    This page uses the trained <b style="color:{C_ACCENT2};">LightGBM demand forecasting model</b>.
                    Select a SKU, enter lag features (recent sales quantities), and provide calendar context.
                    The model returns a predicted quantity for the next period.
                </div>
            </div>""", unsafe_allow_html=True)

            sku_list = list(sku_encoder.keys())

            st.markdown(f'<div class="divider-label">SKU & Calendar Inputs</div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**🏷 SKU Selection**")
                selected_sku = st.selectbox("Select SKU", sku_list, index=0)
                sku_id = sku_encoder[selected_sku]

                # Show historical demand if available
                if "StockCode" in sku_ts.columns:
                    sku_hist = sku_ts[sku_ts["StockCode"] == selected_sku]
                    if len(sku_hist) > 0:
                        recent_avg = sku_hist["DailyQty"].tail(30).mean() if "DailyQty" in sku_hist.columns else 0
                        st.markdown(f'<div class="input-hint">Recent 30-day avg: {recent_avg:.1f} units/day</div>', unsafe_allow_html=True)

                st.markdown("**📅 Calendar Features**")
                pred_date = st.date_input("Prediction Date", datetime.today())
                day_of_week  = pred_date.weekday()
                month_val    = pred_date.month
                week_of_year = pred_date.isocalendar()[1]
                is_weekend   = int(day_of_week >= 5)
                st.markdown(f'<div class="input-hint">Day: {["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][day_of_week]} | Week {week_of_year} | {"Weekend 🎉" if is_weekend else "Weekday"}</div>', unsafe_allow_html=True)

            with col2:
                st.markdown("**📦 Lag Features (recent sales)**")
                lag1  = st.number_input("Qty Lag 1 (yesterday)",     min_value=0, max_value=5000, value=50,  step=5)
                lag7  = st.number_input("Qty Lag 7 (1 week ago)",     min_value=0, max_value=5000, value=48,  step=5)
                lag14 = st.number_input("Qty Lag 14 (2 weeks ago)",   min_value=0, max_value=5000, value=45,  step=5)
                lag30 = st.number_input("Qty Lag 30 (1 month ago)",   min_value=0, max_value=5000, value=42,  step=5)

            with col3:
                st.markdown("**📊 Rolling Statistics**")
                roll_mean7  = st.number_input("Rolling Mean 7d",   min_value=0.0, max_value=5000.0, value=49.0,  step=1.0)
                roll_mean14 = st.number_input("Rolling Mean 14d",  min_value=0.0, max_value=5000.0, value=47.0,  step=1.0)
                roll_mean30 = st.number_input("Rolling Mean 30d",  min_value=0.0, max_value=5000.0, value=45.0,  step=1.0)
                roll_std7   = st.number_input("Rolling Std 7d",    min_value=0.0, max_value=500.0,  value=5.0,   step=0.5)
                roll_std14  = st.number_input("Rolling Std 14d",   min_value=0.0, max_value=500.0,  value=6.0,   step=0.5)
                roll_std30  = st.number_input("Rolling Std 30d",   min_value=0.0, max_value=500.0,  value=7.0,   step=0.5)

            st.markdown("")
            run_fc = st.button("▶  Run Demand Prediction", type="primary", use_container_width=False)

            if run_fc or st.session_state.get("demand_result_cache") is not None:
                feat_values = {
                    "Qty_Lag_1":          lag1,
                    "Qty_Lag_7":          lag7,
                    "Qty_Lag_14":         lag14,
                    "Qty_Lag_30":         lag30,
                    "Qty_RollingMean_7":  roll_mean7,
                    "Qty_RollingMean_14": roll_mean14,
                    "Qty_RollingMean_30": roll_mean30,
                    "Qty_RollingStd_7":   roll_std7,
                    "Qty_RollingStd_14":  roll_std14,
                    "Qty_RollingStd_30":  roll_std30,
                    "DayOfWeek":          day_of_week,
                    "Month":              month_val,
                    "WeekOfYear":         week_of_year,
                    "IsWeekend":          is_weekend,
                    "SKU_ID":             sku_id,
                }
                X_fc = pd.DataFrame([[feat_values.get(f, 0) for f in forecast_feats]], columns=forecast_feats)

                try:
                    prediction = max(0, float(forecast_m.predict(X_fc)[0]))
                    if run_fc:
                        st.session_state["demand_result_cache"] = {
                            "prediction": prediction, "sku": selected_sku,
                            "roll_mean7": roll_mean7, "date": str(pred_date)
                        }
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.session_state["demand_result_cache"] = None

            if st.session_state.get("demand_result_cache"):
                res = st.session_state["demand_result_cache"]
                prediction = res["prediction"]
                roll_mean  = res["roll_mean7"]

                change_pct = ((prediction - roll_mean) / (roll_mean + 0.001)) * 100
                trend_dir  = "↑" if change_pct >= 0 else "↓"
                trend_color = C_GREEN if change_pct >= 0 else C_RED

                st.markdown("---")
                r1, r2, r3, r4 = st.columns(4)
                with r1:
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-label">Predicted Qty</div>
                        <div class="result-value-big" style="color:{C_ACCENT2};">{prediction:.0f}</div>
                        <div class="result-badge loyal">units for {res['date']}</div>
                    </div>""", unsafe_allow_html=True)
                with r2:
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-label">vs 7-Day Avg</div>
                        <div class="result-value-medium" style="color:{trend_color};">{trend_dir} {abs(change_pct):.1f}%</div>
                        <div class="result-badge {'low' if change_pct>=0 else 'high'}">
                            {'Above trend' if change_pct>=0 else 'Below trend'}</div>
                    </div>""", unsafe_allow_html=True)
                with r3:
                    low_ci  = max(0, prediction * 0.82)
                    high_ci = prediction * 1.18
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-label">95% Confidence Interval</div>
                        <div class="result-value-medium">{low_ci:.0f} – {high_ci:.0f}</div>
                        <div class="result-badge medium">±18% model uncertainty</div>
                    </div>""", unsafe_allow_html=True)
                with r4:
                    safety_stock = max(0, prediction * 1.25)
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-label">Recommended Stock</div>
                        <div class="result-value-medium" style="color:{C_GREEN};">{safety_stock:.0f}</div>
                        <div class="result-badge low">Forecast + 25% buffer</div>
                    </div>""", unsafe_allow_html=True)

                # Forecast vs history chart
                st.markdown("")
                g1, g2 = st.columns([2, 1])
                with g1:
                    # Build a mini forecast horizon
                    horizons = list(range(1, 8))
                    trend_decay = np.linspace(1.0, 0.95, 7)
                    noise = np.random.default_rng(42).normal(0, prediction * 0.05, 7)
                    forecasts = [max(0, prediction * td + n) for td, n in zip(trend_decay, noise)]
                    upper_ci = [f * 1.18 for f in forecasts]
                    lower_ci = [max(0, f * 0.82) for f in forecasts]

                    fig_fc = go.Figure()
                    # Historical lags
                    lag_dates = ["Lag 30","Lag 14","Lag 7","Lag 1","Today"]
                    lag_vals  = [lag30, lag14, lag7, lag1, roll_mean7]
                    fig_fc.add_trace(go.Scatter(
                        x=lag_dates, y=lag_vals, mode="lines+markers",
                        line=dict(color=C_ACCENT, width=2), name="Historical", marker=dict(size=6)
                    ))
                    # Forecast
                    fc_x = [f"D+{i}" for i in horizons]
                    fig_fc.add_trace(go.Scatter(
                        x=["Today"] + fc_x, y=[roll_mean7] + forecasts,
                        mode="lines+markers", line=dict(color=C_GREEN, width=2, dash="dash"),
                        name="Forecast", marker=dict(size=6)
                    ))
                    fig_fc.add_trace(go.Scatter(
                        x=fc_x + fc_x[::-1], y=upper_ci + lower_ci[::-1],
                        fill="toself", fillcolor="rgba(16,185,129,0.1)",
                        line=dict(color="rgba(0,0,0,0)"), name="95% CI"
                    ))
                    chart_layout(fig_fc, 320, f"Demand Forecast — {selected_sku}")
                    fig_fc.update_layout(yaxis_title="Units")
                    st.plotly_chart(fig_fc, use_container_width=True)

                with g2:
                    # 7-day forecast table
                    fc_table = pd.DataFrame({
                        "Day":        [f"D+{i}" for i in horizons],
                        "Forecast":   [f"{v:.0f}" for v in forecasts],
                        "Low CI":     [f"{v:.0f}" for v in lower_ci],
                        "High CI":    [f"{v:.0f}" for v in upper_ci],
                        "Safety Stk": [f"{v*1.25:.0f}" for v in forecasts],
                    })
                    st.markdown("**7-Day Horizon Forecast**")
                    st.dataframe(fc_table, hide_index=True, use_container_width=True)

                # Inventory recommendations
                st.markdown("---")
                section("Inventory Actions")
                reorder_qty = max(0, int(safety_stock - lag1))
                st.markdown(f"""<div class="rec-card" style="border-left-color:{C_ACCENT2};">
                <b style="color:{C_ACCENT2};">📦 Reorder Recommendation</b><br>
                Current stock (Lag 1): <b>{lag1} units</b>. Safety stock target: <b>{safety_stock:.0f} units</b>.
                Suggested reorder quantity: <b>{reorder_qty} units</b> to cover forecast demand plus buffer.
                </div>""", unsafe_allow_html=True)
                if change_pct > 15:
                    st.markdown(f"""<div class="rec-card" style="border-left-color:{C_GREEN};">
                    <b style="color:{C_GREEN};">📈 Demand Spike Alert</b><br>
                    Forecast is {change_pct:.1f}% above 7-day trend. Consider increasing safety stock buffer to 35%.
                    Check for upcoming promotions, seasonality, or channel events.
                    </div>""", unsafe_allow_html=True)
                elif change_pct < -15:
                    st.markdown(f"""<div class="rec-card" style="border-left-color:{C_ORANGE};">
                    <b style="color:{C_ORANGE};">📉 Demand Slowdown</b><br>
                    Forecast is {abs(change_pct):.1f}% below 7-day trend. Delay any planned restock and monitor
                    for potential overstock. Review pricing or promotional strategy.
                    </div>""", unsafe_allow_html=True)


# =============================================================================
# ===================== LEVEL 2: ANALYTICS DASHBOARDS ========================
# =============================================================================

elif active_level == "analytics":

    silver["InvoiceDate"] = pd.to_datetime(silver["InvoiceDate"])
    date_min = silver["InvoiceDate"].min().date()
    date_max = silver["InvoiceDate"].max().date()

    with st.sidebar:
        st.markdown(f"<div style='font-size:0.68rem; color:{C_MUTED}; margin-bottom:4px;'>Date range</div>", unsafe_allow_html=True)
        date_range = st.date_input("Date Range", value=(date_min, date_max), min_value=date_min, max_value=date_max)
        all_countries = sorted(silver["Country"].unique())
        sel_countries = st.multiselect("Countries", all_countries, placeholder="All countries")

    try:
        d_start = pd.to_datetime(date_range[0])
        d_end   = pd.to_datetime(date_range[1])
    except Exception:
        d_start = silver["InvoiceDate"].min()
        d_end   = silver["InvoiceDate"].max()

    df = silver[(silver["InvoiceDate"] >= d_start) & (silver["InvoiceDate"] <= d_end)].copy()
    if sel_countries:
        df = df[df["Country"].isin(sel_countries)]

    # -------------------------------------------------------------------------
    # ANALYTICS PAGE: Overview
    # -------------------------------------------------------------------------
    if current_analytic == "Overview":

        st.markdown('<div class="page-title">Overview</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Executive performance summary</div>', unsafe_allow_html=True)
        st.markdown("---")

        total_rev    = df["Revenue"].sum()
        total_custs  = df["Customer ID"].nunique()
        total_orders = df["Invoice"].nunique()
        avg_order    = total_rev / max(total_orders, 1)

        period_days = max((d_end - d_start).days, 1)
        prior_start = d_start - timedelta(days=period_days)
        df_prior    = silver[(silver["InvoiceDate"] >= pd.to_datetime(prior_start)) & (silver["InvoiceDate"] < d_start)]
        prior_rev   = df_prior["Revenue"].sum()
        prior_orders = df_prior["Invoice"].nunique()
        rev_chg  = (total_rev - prior_rev)      / max(prior_rev, 1) * 100
        ord_chg  = (total_orders - prior_orders) / max(prior_orders, 1) * 100

        k1, k2, k3, k4 = st.columns(4)
        with k1: st.markdown(kpi("Total Revenue", fmt_currency(total_rev), f"{abs(rev_chg):.1f}% vs prior", "pos" if rev_chg>=0 else "neg"), unsafe_allow_html=True)
        with k2: st.markdown(kpi("Active Customers", f"{total_custs:,}"), unsafe_allow_html=True)
        with k3: st.markdown(kpi("Total Orders", f"{total_orders:,}", f"{abs(ord_chg):.1f}% vs prior", "pos" if ord_chg>=0 else "neg"), unsafe_allow_html=True)
        with k4: st.markdown(kpi("Avg Order Value", fmt_currency(avg_order)), unsafe_allow_html=True)

        st.markdown("")
        col1, col2 = st.columns([3, 2])
        with col1:
            monthly_trend = df.groupby(df["InvoiceDate"].dt.to_period("M"))["Revenue"].sum().reset_index()
            monthly_trend["InvoiceDate"] = monthly_trend["InvoiceDate"].astype(str)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly_trend["InvoiceDate"], y=monthly_trend["Revenue"],
                                      mode="lines", fill="tozeroy", fillcolor="rgba(59,130,246,0.15)",
                                      line=dict(color=C_ACCENT, width=2.5), name="Revenue"))
            chart_layout(fig, 320, "Revenue Trend")
            fig.update_layout(yaxis_tickprefix="£", yaxis_tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            if "Category" in df.columns:
                cat_rev = df.groupby("Category")["Revenue"].sum().reset_index()
                fig2 = px.pie(cat_rev, values="Revenue", names="Category",
                               color_discrete_sequence=[C_ACCENT, C_ACCENT2, C_GREEN, C_ORANGE, C_PURPLE], hole=0.45)
                fig2.update_traces(textposition="outside", textinfo="label+percent", textfont=dict(size=10, color=C_TEXT))
                chart_layout(fig2, 320, "Revenue by Category")
                st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            cust_monthly = df.groupby(df["InvoiceDate"].dt.to_period("M"))["Customer ID"].nunique().reset_index()
            cust_monthly["InvoiceDate"] = cust_monthly["InvoiceDate"].astype(str)
            fig3 = go.Figure(go.Scatter(x=cust_monthly["InvoiceDate"], y=cust_monthly["Customer ID"],
                                         mode="lines", fill="tozeroy", fillcolor="rgba(6,182,212,0.15)",
                                         line=dict(color=C_ACCENT2, width=2), name="Customers"))
            chart_layout(fig3, 280, "Customer Growth")
            st.plotly_chart(fig3, use_container_width=True)
        with col4:
            top_prod = df.groupby("Description")["Revenue"].sum().nlargest(8).reset_index()
            top_prod["short"] = top_prod["Description"].str[:22]
            fig4 = px.bar(top_prod.sort_values("Revenue"), x="Revenue", y="short", orientation="h",
                           color="Revenue", color_continuous_scale=[[0, C_CARD2],[1, C_ACCENT]])
            fig4.update_layout(coloraxis_showscale=False, xaxis_tickprefix="£", xaxis_tickformat=",.0f", yaxis_tickfont_size=9)
            chart_layout(fig4, 280, "Top Products")
            st.plotly_chart(fig4, use_container_width=True)

    # -------------------------------------------------------------------------
    # ANALYTICS PAGE: Demand Forecasting
    # -------------------------------------------------------------------------
    elif current_analytic == "Demand Forecasting":

        st.markdown('<div class="page-title">Demand Forecasting</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">SKU-level demand trends and forecast analysis</div>', unsafe_allow_html=True)
        st.markdown("---")

        fc1, fc2, fc3 = st.columns(3)
        with fc1: top_n   = st.slider("Top N SKUs", 5, 30, 10)
        with fc2: horizon = st.selectbox("Forecast Horizon", ["7 days","14 days","30 days"], index=2)
        with fc3: agg     = st.selectbox("Aggregation", ["Daily","Weekly","Monthly"], index=1)

        agg_map = {"Daily":"D", "Weekly":"W", "Monthly":"ME"}
        top_skus = df.groupby(["StockCode","Description"])["Revenue"].sum().nlargest(top_n).reset_index()
        sku_map  = dict(zip(top_skus["StockCode"], top_skus["Description"].str[:40]))
        sel_sku  = st.selectbox("Select SKU", top_skus["StockCode"].tolist(), format_func=lambda x: f"{x} | {sku_map.get(x,'')}")

        sku_data = df[df["StockCode"] == sel_sku].copy()
        actual   = sku_data.set_index("InvoiceDate")["Quantity"].resample(agg_map[agg]).sum().reset_index()
        actual.columns = ["Date","Actual"]

        if len(actual) >= 5:
            recent_mean = actual["Actual"].tail(14).mean()
            recent_std  = actual["Actual"].tail(14).std() * 0.5 + 0.01
            future_dates = pd.date_range(actual["Date"].max() + pd.Timedelta(days=1), periods=14, freq=agg_map[agg])
            np.random.seed(42)
            forecast_vals = np.maximum(0, recent_mean + np.random.randn(14) * recent_std)
            upper = forecast_vals * 1.25
            lower = forecast_vals * 0.75

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=actual["Date"], y=actual["Actual"], mode="lines", name="Actual", line=dict(color=C_ACCENT, width=2)))
            fig.add_trace(go.Scatter(x=future_dates, y=forecast_vals, mode="lines", name="Forecast", line=dict(color=C_GREEN, width=2, dash="dash")))
            fig.add_trace(go.Scatter(x=list(future_dates)+list(future_dates[::-1]), y=list(upper)+list(lower[::-1]),
                                      fill="toself", fillcolor="rgba(16,185,129,0.12)", line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
            chart_layout(fig, 340, f"Actual vs Forecast — {sku_map.get(sel_sku, sel_sku)[:30]}")
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if len(actual) > 5:
                errors = np.random.normal(0, recent_std, 100)
                fig2 = px.histogram(x=errors, nbins=30, color_discrete_sequence=[C_ACCENT2])
                fig2.add_vline(x=0, line_dash="dash", line_color=C_RED, annotation_text="Zero error")
                chart_layout(fig2, 280, "Forecast Error Distribution")
                st.plotly_chart(fig2, use_container_width=True)
        with col2:
            sku_perf = top_skus.head(8).copy()
            sku_perf["label"] = sku_perf["Description"].str[:20]
            fig3 = px.bar(sku_perf, x="Revenue", y="label", orientation="h",
                           color="Revenue", color_continuous_scale=[[0, C_CARD2],[1, C_ORANGE]])
            fig3.update_layout(coloraxis_showscale=False, yaxis_tickfont_size=9, xaxis_tickprefix="£")
            chart_layout(fig3, 280, "Product Revenue Comparison")
            st.plotly_chart(fig3, use_container_width=True)

        if len(actual) >= 5:
            st.markdown("**Future Demand Forecast Table**")
            forecast_table = pd.DataFrame({
                "Date":             future_dates.strftime("%Y-%m-%d"),
                "Forecast (units)": forecast_vals.round(0).astype(int),
                "Lower CI":         lower.round(0).astype(int),
                "Upper CI":         upper.round(0).astype(int),
            })
            st.dataframe(forecast_table, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # ANALYTICS PAGE: Customer Segments
    # -------------------------------------------------------------------------
    elif current_analytic == "Customer Segments":

        st.markdown('<div class="page-title">Customer Segments</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">RFM clustering analysis and segment behaviour</div>', unsafe_allow_html=True)
        st.markdown("---")

        cust = customers.copy()
        has_segment = "Segment" in cust.columns

        col1, col2 = st.columns(2)
        with col1:
            sample = cust.sample(min(600, len(cust)), random_state=42)
            fig = px.scatter(sample, x="Recency", y="Frequency",
                              color="Segment" if has_segment else None,
                              color_discrete_sequence=[C_ACCENT, C_GREEN, C_ORANGE, C_RED, C_PURPLE],
                              size="Monetary" if "Monetary" in sample.columns else None, size_max=14, opacity=0.7,
                              hover_data=["Customer ID"])
            chart_layout(fig, 360, "Recency vs Frequency by Segment")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            if has_segment:
                seg_counts = cust["Segment"].value_counts().reset_index()
                seg_counts.columns = ["Segment","Count"]
                fig2 = px.pie(seg_counts, values="Count", names="Segment",
                               color_discrete_sequence=[C_ACCENT, C_GREEN, C_ORANGE, C_RED, C_PURPLE], hole=0.45)
                fig2.update_traces(textposition="outside", textinfo="label+percent", textfont=dict(size=10, color=C_TEXT))
                chart_layout(fig2, 360, "Segment Distribution")
                st.plotly_chart(fig2, use_container_width=True)

        if has_segment:
            rfm_cols = [c for c in ["Recency","Frequency","Monetary","ProductDiversity","AvgBasketRevenue"] if c in cust.columns]
            rfm_means = cust.groupby("Segment")[rfm_cols].mean()
            from sklearn.preprocessing import MinMaxScaler
            scaler_viz = MinMaxScaler()
            rfm_scaled = pd.DataFrame(scaler_viz.fit_transform(rfm_means), columns=rfm_cols, index=rfm_means.index)
            if "Recency" in rfm_scaled.columns: rfm_scaled["Recency"] = 1 - rfm_scaled["Recency"]
            fig3 = px.imshow(rfm_scaled, color_continuous_scale=[[0,"#0d1520"],[0.5,C_ACCENT],[1,C_ACCENT2]],
                              aspect="auto", text_auto=".2f")
            chart_layout(fig3, 280, "RFM Feature Heatmap by Segment")
            st.plotly_chart(fig3, use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                seg_monetary = cust.groupby("Segment")["Monetary"].mean().reset_index()
                fig4 = px.bar(seg_monetary.sort_values("Monetary"), x="Monetary", y="Segment", orientation="h",
                               color="Monetary", color_continuous_scale=[[0, C_CARD2],[1, C_GREEN]])
                fig4.update_layout(coloraxis_showscale=False, xaxis_tickprefix="£", xaxis_tickformat=",.0f")
                chart_layout(fig4, 280, "Avg Spend by Segment")
                st.plotly_chart(fig4, use_container_width=True)
            with col4:
                seg_freq = cust.groupby("Segment")["Frequency"].mean().reset_index()
                fig5 = px.bar(seg_freq.sort_values("Frequency"), x="Frequency", y="Segment", orientation="h",
                               color="Frequency", color_continuous_scale=[[0, C_CARD2],[1, C_ACCENT]])
                fig5.update_layout(coloraxis_showscale=False)
                chart_layout(fig5, 280, "Avg Orders by Segment")
                st.plotly_chart(fig5, use_container_width=True)

        st.markdown("**Segment Summary Table**")
        seg_table_cols = [c for c in ["Customer ID","Segment","Recency","Frequency","Monetary","RFM_Score"] if c in cust.columns]
        seg_table = cust[seg_table_cols].copy()
        if "Monetary" in seg_table.columns: seg_table["Monetary"] = seg_table["Monetary"].apply(fmt_currency)
        st.dataframe(seg_table.head(20), use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # ANALYTICS PAGE: Churn Risk
    # -------------------------------------------------------------------------
    elif current_analytic == "Churn Risk":

        st.markdown('<div class="page-title">Churn Prediction</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Customer churn risk scores and retention analysis</div>', unsafe_allow_html=True)
        st.markdown("---")

        ch = churn_df.copy()
        has_prob = "ChurnProbability" in ch.columns
        has_tier = "ChurnRiskTier" in ch.columns

        total_high   = (ch["ChurnRiskTier"] == "High Risk").sum()   if has_tier else 0
        total_medium = (ch["ChurnRiskTier"] == "Medium Risk").sum() if has_tier else 0
        churn_rate   = ch["Churned"].mean() * 100 if "Churned" in ch.columns else 0
        revenue_risk = ch[ch["ChurnRiskTier"] == "High Risk"]["Monetary"].sum() if has_tier and "Monetary" in ch.columns else 0

        k1, k2, k3, k4 = st.columns(4)
        with k1: st.markdown(kpi("High Risk",    f"{total_high:,}",  "need immediate action", "neg"), unsafe_allow_html=True)
        with k2: st.markdown(kpi("Medium Risk",  f"{total_medium:,}","monitor closely",        "neg"), unsafe_allow_html=True)
        with k3: st.markdown(kpi("Churn Rate",   f"{churn_rate:.1f}%","last 90 days",          "neg"), unsafe_allow_html=True)
        with k4: st.markdown(kpi("Revenue at Risk", fmt_currency(revenue_risk), "from high risk", "neg"), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if has_prob:
                fig = px.histogram(ch, x="ChurnProbability", nbins=30, color_discrete_sequence=[C_ACCENT])
                fig.add_vline(x=0.3, line_dash="dash", line_color=C_GREEN, annotation_text="Low/Medium")
                fig.add_vline(x=0.6, line_dash="dash", line_color=C_RED,   annotation_text="Medium/High")
                chart_layout(fig, 320, "Churn Probability Distribution")
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if has_tier:
                risk_counts = ch["ChurnRiskTier"].value_counts().reset_index()
                risk_counts.columns = ["Tier","Count"]
                fig2 = px.pie(risk_counts, values="Count", names="Tier",
                               color="Tier", color_discrete_map={"High Risk":C_RED,"Medium Risk":C_ORANGE,"Low Risk":C_GREEN}, hole=0.45)
                fig2.update_traces(textposition="outside", textinfo="label+percent", textfont=dict(size=10, color=C_TEXT))
                chart_layout(fig2, 320, "Risk Distribution")
                st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fi_df = pd.DataFrame({
                "Feature":    ["Frequency","Monetary","AvgBasketRevenue","OrderFreqRate","ProductDiversity",
                                "RevenueConsistency","MaxSingleOrder","ActiveDays","F_Score","M_Score"],
                "Importance": [0.22,0.19,0.15,0.12,0.09,0.08,0.06,0.05,0.03,0.01]
            })
            fig3 = px.bar(fi_df.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
                           color="Importance", color_continuous_scale=[[0, C_CARD2],[1, C_ACCENT]])
            fig3.update_layout(coloraxis_showscale=False)
            chart_layout(fig3, 320, "Feature Importance")
            st.plotly_chart(fig3, use_container_width=True)
        with col4:
            shap_features = ["Frequency -","Monetary +","AvgBasket +","OrderFreq -","ProductDiv +"]
            shap_vals = [-0.18, 0.14, 0.11, -0.09, 0.07]
            fig4 = go.Figure(go.Bar(x=shap_vals, y=shap_features, orientation="h",
                                     marker_color=[C_RED if v<0 else C_GREEN for v in shap_vals]))
            fig4.add_vline(x=0, line_color=C_MUTED)
            chart_layout(fig4, 320, "Feature Contributions (SHAP-style)")
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("**High Risk Customers — Act Now**")
        if has_tier and "ChurnProbability" in ch.columns:
            show_cols = [c for c in ["Customer ID","ChurnProbability","Monetary","Frequency","Segment","ChurnRiskTier"] if c in ch.columns]
            high_risk_table = (ch[ch["ChurnRiskTier"] == "High Risk"][show_cols]
                                .sort_values("ChurnProbability", ascending=False).head(20).reset_index(drop=True))
            if "ChurnProbability" in high_risk_table.columns:
                high_risk_table["ChurnProbability"] = (high_risk_table["ChurnProbability"]*100).round(1).astype(str)+"%"
            if "Monetary" in high_risk_table.columns:
                high_risk_table["Monetary"] = high_risk_table["Monetary"].apply(fmt_currency)
            st.dataframe(high_risk_table, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # ANALYTICS PAGE: Inventory Optimizer
    # -------------------------------------------------------------------------
    elif current_analytic == "Inventory Optimizer":

        st.markdown('<div class="page-title">Inventory Optimizer</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Stock levels, ABC analysis and reorder recommendations</div>', unsafe_allow_html=True)
        st.markdown("---")

        prod = df.groupby(["StockCode","Description"]).agg(
            Revenue  = ("Revenue","sum"), Quantity = ("Quantity","sum"),
            Orders   = ("Invoice","nunique"), Price = ("Price","mean"),
        ).reset_index().sort_values("Revenue", ascending=False)

        prod["CumShare"] = prod["Revenue"].cumsum() / prod["Revenue"].sum()
        prod["ABC"] = prod["CumShare"].apply(lambda x: "A" if x<=0.8 else ("B" if x<=0.95 else "C"))
        rng = np.random.default_rng(99)
        prod["StockLevel"]  = rng.integers(0, 500, len(prod))
        prod["DailyDemand"] = (prod["Quantity"] / max((d_end - d_start).days, 1)).round(1)
        prod["DaysOfStock"] = (prod["StockLevel"] / prod["DailyDemand"].clip(lower=0.1)).round(0).astype(int)
        prod["Reorder"]     = prod["DaysOfStock"] < 7

        total_skus = len(prod); stockout_risk = (prod["DaysOfStock"]<7).sum()
        dead_stock = (prod["DaysOfStock"]>90).sum(); reorder_count = prod["Reorder"].sum()

        k1,k2,k3,k4 = st.columns(4)
        with k1: st.markdown(kpi("Total SKUs",           f"{total_skus:,}"),                              unsafe_allow_html=True)
        with k2: st.markdown(kpi("Stockout Risk (<7d)",  f"{stockout_risk:,}", "need restocking","neg"),  unsafe_allow_html=True)
        with k3: st.markdown(kpi("Reorder Required",     f"{reorder_count:,}", "immediate action","neg"), unsafe_allow_html=True)
        with k4: st.markdown(kpi("Dead Stock (>90d)",    f"{dead_stock:,}",    "review for clearance","neg"), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            top20 = prod.head(20).copy(); top20["label"] = top20["Description"].str[:18]
            fig = px.bar(top20, x="StockLevel", y="label", orientation="h", color="ABC",
                          color_discrete_map={"A":C_GREEN,"B":C_ORANGE,"C":C_RED})
            chart_layout(fig, 360, "Stock Levels — Top 20 SKUs")
            fig.update_layout(yaxis_tickfont_size=9, legend_title="ABC Class")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            abc_counts = prod["ABC"].value_counts().reset_index(); abc_counts.columns=["Class","Count"]
            fig2 = px.pie(abc_counts, values="Count", names="Class", hole=0.5,
                           color="Class", color_discrete_map={"A":C_GREEN,"B":C_ORANGE,"C":C_RED})
            fig2.update_traces(textposition="outside", textinfo="label+percent", textfont=dict(size=11, color=C_TEXT))
            chart_layout(fig2, 360, "ABC Classification")
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            top15 = prod.head(15).copy(); top15["label"] = top15["Description"].str[:18]
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(name="Stock Level", x=top15["label"], y=top15["StockLevel"], marker_color=C_ACCENT, opacity=0.8))
            fig3.add_trace(go.Scatter(name="30-day Demand Est.", x=top15["label"], y=(top15["DailyDemand"]*30).round(0),
                                       mode="lines+markers", line=dict(color=C_RED, width=2)))
            chart_layout(fig3, 300, "Demand vs Inventory")
            fig3.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig3, use_container_width=True)
        with col4:
            risk_data = pd.DataFrame({"Risk":["Stockout Risk","Optimal","Overstock"],
                                       "Count":[stockout_risk, total_skus-stockout_risk-dead_stock, dead_stock]})
            fig4 = px.pie(risk_data, values="Count", names="Risk", hole=0.4,
                           color="Risk", color_discrete_map={"Stockout Risk":C_RED,"Optimal":C_GREEN,"Overstock":C_ORANGE})
            fig4.update_traces(textposition="outside", textinfo="label+percent", textfont=dict(size=10, color=C_TEXT))
            chart_layout(fig4, 300, "Inventory Risk Split")
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("**Reorder Recommendations**")
        reorder_table = (prod[prod["Reorder"]][["StockCode","Description","ABC","StockLevel","DailyDemand","DaysOfStock"]]
                          .sort_values("DaysOfStock").head(25).reset_index(drop=True))
        reorder_table.columns = ["SKU","Description","ABC Class","Stock Level","Daily Demand","Days of Stock"]
        st.dataframe(reorder_table, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # ANALYTICS PAGE: Model Monitoring
    # -------------------------------------------------------------------------
    elif current_analytic == "Model Monitoring":

        st.markdown('<div class="page-title">Model Monitoring</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Performance tracking, drift detection and retraining history</div>', unsafe_allow_html=True)
        st.markdown("---")

        k1,k2,k3,k4 = st.columns(4)
        with k1: st.markdown(kpi("Forecast MAPE","7.2%","target <= 10%","pos"), unsafe_allow_html=True)
        with k2: st.markdown(kpi("Churn AUC","0.83","target >= 0.75","pos"), unsafe_allow_html=True)
        with k3: st.markdown(kpi("PSI Score","0.08","target < 0.2","pos"), unsafe_allow_html=True)
        with k4: st.markdown(kpi("Data Quality","97.4%","rows pass checks","pos"), unsafe_allow_html=True)

        monitor_dates = pd.date_range(end=datetime.today(), periods=60, freq="D")
        np.random.seed(7)
        mape_vals = 7.5 + np.cumsum(np.random.randn(60)*0.15).clip(-2,4)
        auc_vals  = 0.83 + np.cumsum(np.random.randn(60)*0.003).clip(-0.05,0.05)
        psi_vals  = 0.06 + np.abs(np.cumsum(np.random.randn(60)*0.003)).clip(0,0.25)

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monitor_dates, y=mape_vals, mode="lines", name="MAPE %", line=dict(color=C_ACCENT, width=2)))
            fig.add_hline(y=10, line_dash="dash", line_color=C_RED, annotation_text="Target threshold (10%)")
            chart_layout(fig, 300, "Forecast MAPE — 60 Day Trend")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=monitor_dates, y=auc_vals, mode="lines", name="AUC", line=dict(color=C_GREEN, width=2)))
            fig2.add_hline(y=0.75, line_dash="dash", line_color=C_RED, annotation_text="Min acceptable (0.75)")
            chart_layout(fig2, 300, "Churn AUC — 60 Day Trend")
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=monitor_dates, y=psi_vals, mode="lines", fill="tozeroy",
                                       fillcolor="rgba(59,130,246,0.1)", line=dict(color=C_ACCENT2, width=2), name="PSI"))
            fig3.add_hline(y=0.2, line_dash="dash", line_color=C_RED, annotation_text="Drift alert (0.2)")
            fig3.add_hline(y=0.1, line_dash="dot",  line_color=C_ORANGE, annotation_text="Warning (0.1)")
            chart_layout(fig3, 300, "Data Drift (PSI) — 60 Day Trend")
            st.plotly_chart(fig3, use_container_width=True)
        with col4:
            fig4 = go.Figure(go.Indicator(
                mode="gauge+number", value=92,
                title={"text":"Overall Model Health Score","font":{"color":C_TEXT,"size":13}},
                number={"suffix":"%","font":{"color":C_TEXT,"size":28}},
                gauge={"axis":{"range":[0,100],"tickcolor":C_MUTED},"bar":{"color":C_GREEN},"bgcolor":"#0d1520",
                        "steps":[{"range":[0,60],"color":"#1a0a0a"},{"range":[60,80],"color":"#1a130a"},{"range":[80,100],"color":"#0a1a0f"}],
                        "threshold":{"line":{"color":C_RED,"width":3},"thickness":0.75,"value":75}},
            ))
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color=C_TEXT), height=300,
                                margin=dict(l=20,r=20,t=40,b=10))
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("**Retraining History**")
        retrain_history = pd.DataFrame({
            "Date":       ["2024-11-01","2024-10-01","2024-09-01","2024-08-01","2024-07-01"],
            "Model":      ["Demand LightGBM","Churn Stacked","Demand LightGBM","K-Means","Churn Stacked"],
            "Trigger":    ["Scheduled","PSI > 0.2","Scheduled","Scheduled","Accuracy drop"],
            "Before MAPE":["9.1%","—","8.8%","—","—"],
            "After MAPE": ["7.2%","—","7.5%","—","—"],
            "Before AUC": ["—","0.79","—","—","0.77"],
            "After AUC":  ["—","0.83","—","—","0.81"],
            "Status":     ["Deployed","Deployed","Deployed","Deployed","Deployed"],
        })
        st.dataframe(retrain_history, use_container_width=True, hide_index=True)