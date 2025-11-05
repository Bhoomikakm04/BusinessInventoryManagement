
"""
Business Inventory Dashboard - v4 (Demo-ready + improved UI)

Features added:
 - Demo CSVs included in /mnt/data (Load demo fills from these files)
 - Company logo in sidebar
 - Interactive filters: date range, product search
 - Exportable charts: download Sales Trend as PNG, and cleaned CSVs
 - Centralized CSS "theme" via injected styles (for this demo)
"""

import streamlit as st
import pandas as pd, numpy as np, os
from datetime import datetime
import altair as alt
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="AIC Inventory Dashboard v4", layout="wide", initial_sidebar_state="expanded")

# Paths (demo files are created by the environment)
PERSIST_SALES = "/mnt/data/last_sales.csv"
PERSIST_STOCK = "/mnt/data/last_stock.csv"
PERSIST_REVIEWS = "/mnt/data/last_reviews.csv"
DEMO_SALES = "/mnt/data/demo_sales_dataset.csv"
DEMO_STOCK = "/mnt/data/demo_stock_dataset.csv"
DEMO_REVIEWS = "/mnt/data/demo_reviews_dataset.csv"
LOGO = "/mnt/data/aic_logo.png"

# --- Theme / CSS ---
st.markdown(\"\"\"
    <style>
    :root{
        --accent:#0f3b47;
        --muted:#6b7280;
        --card-bg:linear-gradient(180deg,#ffffff,#f6f8fb);
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] { background-color: var(--accent); color: white; }
    [data-testid="stSidebar"] .css-1lcbmhc { color: white; }
    /* KPI card */
    .kpi-card {background: var(--card-bg); border-radius:12px; padding:12px; box-shadow:0 2px 8px rgba(0,0,0,0.06);}
    .kpi-title{color:var(--muted); font-size:13px;}
    .kpi-value{font-weight:700; font-size:20px;}
    .small-muted{color:var(--muted); font-size:12px;}
    </style>
\"\"\", unsafe_allow_html=True)

# --- Sidebar content ---
with st.sidebar:
    st.image(LOGO, width=120)
    st.title("AIC Inventory App")
    nav = st.radio("", ("Dashboard", "Inventory", "Reports", "Settings"))
    st.markdown("---")
    st.header("Data uploads")
    sales_file = st.file_uploader("Sales CSV", type=["csv"], key="s4")
    stock_file = st.file_uploader("Stock CSV", type=["csv"], key="st4")
    reviews_file = st.file_uploader("Reviews CSV", type=["csv"], key="r4")
    cols = st.columns(2)
    load_demo = cols[0].button("Load demo")
    use_last = cols[1].checkbox("Use last", value=False)
    st.markdown("---")
    st.header("Filters (global)")
    # date range and product search will be populated after data loads
    st.markdown("Use the dashboard filters to the right after loading data.")

# Persist uploads if provided
def save_persisted(uploaded, path):
    if uploaded is None: return False
    if hasattr(uploaded, "read"):
        uploaded.seek(0)
        with open(path, "wb") as f:
            f.write(uploaded.read())
        return True
    return False

if sales_file is not None: save_persisted(sales_file, PERSIST_SALES)
if stock_file is not None: save_persisted(stock_file, PERSIST_STOCK)
if reviews_file is not None: save_persisted(reviews_file, PERSIST_REVIEWS)

if use_last:
    if os.path.exists(PERSIST_SALES): sales_file = PERSIST_SALES
    if os.path.exists(PERSIST_STOCK): stock_file = PERSIST_STOCK
    if os.path.exists(PERSIST_REVIEWS): reviews_file = PERSIST_REVIEWS

if load_demo:
    if os.path.exists(DEMO_SALES): sales_file = DEMO_SALES
    if os.path.exists(DEMO_STOCK): stock_file = DEMO_STOCK
    if os.path.exists(DEMO_REVIEWS): reviews_file = DEMO_REVIEWS

# --- Helpers ---
@st.cache_data
def safe_read(path_or_buf):
    try:
        if hasattr(path_or_buf, "read"):
            path_or_buf.seek(0)
            df = pd.read_csv(path_or_buf)
        else:
            df = pd.read_csv(path_or_buf)
        return df, None
    except Exception as e:
        try:
            if hasattr(path_or_buf, "read"):
                path_or_buf.seek(0)
            df = pd.read_csv(path_or_buf, encoding="ISO-8859-1")
            return df, "ISO-8859-1"
        except Exception as e2:
            return None, str(e2)

def normalize_sales_df(df):
    df = df.copy()
    if "Date" not in df.columns:
        for c in df.columns:
            if "date" in c.lower():
                df = df.rename(columns={c:"Date"})
                break
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Product" not in df.columns:
        for c in df.columns:
            if "product" in c.lower():
                df = df.rename(columns={c:"Product"})
                break
    if "Sales" not in df.columns:
        numcols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numcols:
            df = df.rename(columns={numcols[0]:"Sales"})
        else:
            for c in df.columns:
                if any(k in c.lower() for k in ["qty","quantity","units","amount","sales"]):
                    df = df.rename(columns={c:"Sales"})
                    break
    if "Sales" not in df.columns:
        raise ValueError("No numeric Sales column found.")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
    if "Country" not in df.columns:
        df["Country"] = "All"
    return df

# --- Load data ---
if sales_file is None:
    st.info("Upload sales CSV, click 'Load demo' or 'Use last' to populate the dashboard.")
    st.stop()

sales_df, err = safe_read(sales_file)
if sales_df is None:
    st.error(f"Failed to read sales CSV: {err}")
    st.stop()
try:
    sales_df = normalize_sales_df(sales_df)
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

stock_df = None
if stock_file is not None:
    stock_df, _ = safe_read(stock_file)
reviews_df = None
if reviews_file is not None:
    reviews_df, _ = safe_read(reviews_file)

# --- Interactive filters ---
min_date = sales_df['Date'].min()
max_date = sales_df['Date'].max()
colf1, colf2 = st.columns([3,1])
with colf1:
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
with colf2:
    product_search = st.selectbox("Product", options=["All"] + sorted(sales_df['Product'].unique().tolist()))

# apply filters
sdf = sales_df.copy()
start, end = date_range
sdf = sdf[(sdf['Date'] >= pd.to_datetime(start)) & (sdf['Date'] <= pd.to_datetime(end))]
if product_search != "All":
    sdf = sdf[sdf['Product'] == product_search]

# --- KPIs ---
total_sales = sdf['Sales'].sum()
total_orders = len(sdf)
avg_daily = sdf.groupby('Date')['Sales'].sum().mean()

k1,k2,k3 = st.columns(3)
k1.markdown(f\"\"\"<div class='kpi-card'><div class='kpi-title'>Total Sales</div><div class='kpi-value'>${total_sales:,.0f}</div><div class='small-muted'>Filtered</div></div>\"\"\", unsafe_allow_html=True)
k2.markdown(f\"\"\"<div class='kpi-card'><div class='kpi-title'>Avg Daily</div><div class='kpi-value'>{avg_daily:.2f}</div><div class='small-muted'>Over date range</div></div>\"\"\", unsafe_allow_html=True)
k3.markdown(f\"\"\"<div class='kpi-card'><div class='kpi-title'>Orders</div><div class='kpi-value'>{total_orders}</div><div class='small-muted'>Records</div></div>\"\"\", unsafe_allow_html=True)

st.markdown(\"---\")

# --- Charts ---
left, right = st.columns([3,1])
with left:
    st.subheader("Sales Trend")
    daily = sdf.groupby('Date')['Sales'].sum().reset_index()
    chart = alt.Chart(daily).mark_area(opacity=0.7).encode(x='Date:T', y='Sales:Q').properties(height=320)
    st.altair_chart(chart, use_container_width=True)

    # Exportable chart as PNG (matplotlib)
    buf = BytesIO()
    fig, ax = plt.subplots(figsize=(10,3))
    ax.fill_between(daily['Date'], daily['Sales'], alpha=0.3)
    ax.plot(daily['Date'], daily['Sales'])
    ax.set_title("Sales Trend")
    ax.set_ylabel("Sales")
    fig.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    st.download_button("Download Sales Trend (PNG)", data=buf, file_name="sales_trend.png", mime="image/png")

with right:
    st.subheader("Units left (low stock)")
    if stock_df is None:
        st.info("Upload stock CSV to view low-stock items.")
    else:
        sd = stock_df.copy()
        if 'Stock' not in sd.columns:
            for c in sd.columns:
                if 'stock' in c.lower(): sd = sd.rename(columns={c:'Stock'}); break
        if 'Product' not in sd.columns:
            for c in sd.columns:
                if 'product' in c.lower(): sd = sd.rename(columns={c:'Product'}); break
        if 'Product' in sd.columns and 'Stock' in sd.columns:
            sd['Stock'] = pd.to_numeric(sd['Stock'], errors='coerce').fillna(0)
            avg7 = sales_df.groupby('Product')['Sales'].apply(lambda x: x.tail(7).mean() if len(x)>=3 else x.mean()).reset_index().rename(columns={'Sales':'Avg7'})
            merged = pd.merge(sd, avg7, on='Product', how='left')
            merged['DaysLeft'] = merged['Stock'] / merged['Avg7'].replace(0, np.nan)
            low = merged.sort_values('DaysLeft').head(8)
            st.table(low[['Product','Stock','Avg7','DaysLeft']].fillna('N/A'))
        else:
            st.write('Stock missing Product/Stock cols')

st.markdown('---')
# --- Best sellers & export ---
st.subheader("Best sellers")
best = sdf.groupby('Product')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
st.table(best.head(10))
st.download_button('Download filtered sales (CSV)', data=sdf.to_csv(index=False).encode('utf-8'), file_name='sales_filtered.csv', mime='text/csv')
if stock_df is not None:
    st.download_button('Download stock CSV', data=stock_df.to_csv(index=False).encode('utf-8'), file_name='stock.csv', mime='text/csv')
if reviews_df is not None:
    st.download_button('Download reviews CSV', data=reviews_df.to_csv(index=False).encode('utf-8'), file_name='reviews.csv', mime='text/csv')

st.caption('Tip: To customize app theme persistently, create ~/.streamlit/config.toml with a [theme] section.')
