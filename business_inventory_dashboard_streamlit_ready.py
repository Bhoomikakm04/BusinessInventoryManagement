
"""
Business Inventory Dashboard - Streamlit Cloud ready

Notes for Streamlit Community Cloud:
 - Do NOT rely on /mnt/data persistence on Streamlit Cloud; instead commit demo CSVs into the repository under a "data/" folder.
 - Read demo files from relative paths (./data/demo_sales_dataset.csv, etc.). These files must be present in the GitHub repo to be used by "Load demo".
 - Use st.secrets for sensitive keys (OpenAI) rather than typing them into the UI.
 - This app avoids writing to disk for persistence; "Use last" is not supported on Streamlit Cloud due to ephemeral filesystem.
"""

import streamlit as st
import pandas as pd, numpy as np, os
from datetime import datetime
import altair as alt
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="AIC Inventory (Cloud-ready)", layout="wide", initial_sidebar_state="expanded")

# Paths used when demo files are committed to the repository under a 'data' folder
REPO_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEMO_SALES = os.path.join(REPO_DATA_DIR, "demo_sales_dataset.csv")
DEMO_STOCK = os.path.join(REPO_DATA_DIR, "demo_stock_dataset.csv")
DEMO_REVIEWS = os.path.join(REPO_DATA_DIR, "demo_reviews_dataset.csv")
LOGO_PATH = os.path.join(REPO_DATA_DIR, "aic_logo.png")  # commit logo under data/

# --- Sidebar ---
with st.sidebar:
    st.markdown("## AIC Inventory")
    # show logo if included in repo/data
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=120)
    nav = st.radio("", ("Dashboard", "Inventory", "Reports", "Settings"))
    st.markdown("---")
    st.header("Upload or use demo")
    sales_upload = st.file_uploader("Sales CSV", type=["csv"], key="sales_up")
    stock_upload = st.file_uploader("Stock CSV", type=["csv"], key="stock_up")
    reviews_upload = st.file_uploader("Reviews CSV (optional)", type=["csv"], key="reviews_up")
    cols = st.columns(2)
    load_demo = cols[0].button("Load demo (from repo)")
    use_last = cols[1].checkbox("Use last (disabled on Cloud)", value=False, disabled=True)
    st.markdown("---")
    st.caption("On Streamlit Cloud: include demo files in the GitHub repo under the 'data/' folder and deploy. Use st.secrets for API keys.")

# If user uploaded files in the UI, prefer those. Otherwise, if Load demo pressed, try to use demo files from repo.
def which_file(uploaded, demo_path):
    if uploaded is not None:
        return uploaded
    if load_demo and os.path.exists(demo_path):
        return demo_path
    return None

sales_file = which_file(sales_upload, DEMO_SALES)
stock_file = which_file(stock_upload, DEMO_STOCK)
reviews_file = which_file(reviews_upload, DEMO_REVIEWS)

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

# --- Load data or prompt ---
if sales_file is None:
    st.info("Upload sales CSV or commit demo files to the repository under 'data/' and click 'Load demo (from repo)'.")
    st.stop()

sales_df, err = safe_read(sales_file)
if sales_df is None:
    st.error(f"Failed to read sales CSV: {err}")
    st.stop()
try:
    sales_df = normalize_sales_df(sales_df)
except Exception as e:
    st.error(f"Error normalizing sales CSV: {e}")
    st.stop()

stock_df = None
if stock_file is not None:
    stock_df, _ = safe_read(stock_file)
reviews_df = None
if reviews_file is not None:
    reviews_df, _ = safe_read(reviews_file)

# --- Interactive UI (filters) ---
min_date = sales_df['Date'].min()
max_date = sales_df['Date'].max()
date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
product_options = ["All"] + sorted(sales_df['Product'].unique().tolist())
product_filter = st.selectbox("Product", product_options)

# Apply filters
sdf = sales_df.copy()
start, end = date_range
sdf = sdf[(sdf['Date'] >= pd.to_datetime(start)) & (sdf['Date'] <= pd.to_datetime(end))]
if product_filter != "All":
    sdf = sdf[sdf['Product'] == product_filter]

# --- KPIs and charts ---
st.title("AIC Inventory Dashboard (Cloud-ready)")
k1,k2,k3 = st.columns(3)
k1.metric("Total sales", f"${sdf['Sales'].sum():,.0f}")
k2.metric("Avg daily", f"{sdf.groupby('Date')['Sales'].sum().mean():.2f}")
k3.metric("Records", len(sdf))

st.markdown("---")
left, right = st.columns([3,1])
with left:
    st.subheader("Sales trend")
    daily = sdf.groupby('Date')['Sales'].sum().reset_index()
    chart = alt.Chart(daily).mark_area(opacity=0.7).encode(x='Date:T', y='Sales:Q').properties(height=320)
    st.altair_chart(chart, use_container_width=True)
    # export PNG
    buf = BytesIO()
    fig, ax = plt.subplots(figsize=(10,3))
    ax.fill_between(daily['Date'], daily['Sales'], alpha=0.3)
    ax.plot(daily['Date'], daily['Sales'])
    ax.set_title("Sales Trend")
    fig.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    st.download_button("Download Sales Trend (PNG)", data=buf, file_name="sales_trend.png", mime="image/png")

with right:
    st.subheader("Best sellers")
    best = sdf.groupby('Product')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
    st.table(best.head(8))
    st.subheader("Units left (low stock)")
    if stock_df is None:
        st.info("Upload stock CSV to view low-stock items or include demo_stock_dataset.csv in repo/data/")
    else:
        sd = stock_df.copy()
        if 'Stock' not in sd.columns:
            for c in sd.columns:
                if 'stock' in c.lower():
                    sd = sd.rename(columns={c:'Stock'}); break
        if 'Product' not in sd.columns:
            for c in sd.columns:
                if 'product' in c.lower():
                    sd = sd.rename(columns={c:'Product'}); break
        if 'Product' in sd.columns and 'Stock' in sd.columns:
            sd['Stock'] = pd.to_numeric(sd['Stock'], errors='coerce').fillna(0)
            avg7 = sales_df.groupby('Product')['Sales'].apply(lambda x: x.tail(7).mean() if len(x)>=3 else x.mean()).reset_index().rename(columns={'Sales':'Avg7'})
            merged = pd.merge(sd, avg7, on='Product', how='left')
            merged['DaysLeft'] = merged['Stock'] / merged['Avg7'].replace(0, np.nan)
            st.table(merged.sort_values('DaysLeft').head(8)[['Product','Stock','Avg7','DaysLeft']].fillna('N/A'))
        else:
            st.write("Stock file present but missing Product/Stock columns.")

st.markdown("---")
st.download_button("Download filtered sales CSV", data=sdf.to_csv(index=False).encode('utf-8'), file_name='sales_filtered.csv', mime='text/csv')
if stock_df is not None:
    st.download_button("Download stock CSV", data=stock_df.to_csv(index=False).encode('utf-8'), file_name='stock.csv', mime='text/csv')
if reviews_df is not None:
    st.download_button("Download reviews CSV", data=reviews_df.to_csv(index=False).encode('utf-8'), file_name='reviews.csv', mime='text/csv')

st.caption("To deploy on Streamlit Community Cloud: push this repo (including a 'data/' folder with demo CSVs and logo) to GitHub, then create a new app on share.streamlit.io pointing to this file.")
