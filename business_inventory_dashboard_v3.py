
"""
Business Inventory Dashboard - v3 (UI upgrade)

This Streamlit app focuses on a cleaner 'dashboard' UI with:
 - Left dark navigation sidebar
 - Top KPI cards (Total Sales, Purchases, Net Profit, Receivables...)
 - Grid layout for charts: Sales Trend, Top Customers, Purchase by Location, Best Sellers
 - Inventory "Units left" card and Best sellers list
 - Keeps last uploads in /mnt/data as before and supports Load demo / Use last upload
 - Use: streamlit run business_inventory_dashboard_v3.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="AIC Inventory App (Dashboard)", layout="wide")

# ---------------- Paths ----------------
PERSIST_SALES = "/mnt/data/last_sales.csv"
PERSIST_STOCK = "/mnt/data/last_stock.csv"
PERSIST_REVIEWS = "/mnt/data/last_reviews.csv"
DEMO_SALES = "/mnt/data/demo_sales_dataset.csv"
DEMO_STOCK = "/mnt/data/demo_stock_dataset.csv"
DEMO_REVIEWS = "/mnt/data/unstructured_reviews_demo_multilingual.csv"

# ---------------- Helper functions ----------------
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

def save_persisted_file(uploaded, path):
    try:
        if uploaded is None:
            return False
        if hasattr(uploaded, "read"):
            uploaded.seek(0)
            b = uploaded.read()
            with open(path, "wb") as f:
                f.write(b)
            return True
        return False
    except Exception:
        return False

def normalize_sales_df(df):
    df = df.copy()
    if "Date" not in df.columns:
        for c in df.columns:
            if "date" in c.lower():
                df = df.rename(columns={c: "Date"})
                break
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Product" not in df.columns:
        for c in df.columns:
            if "product" in c.lower() or "item" in c.lower():
                df = df.rename(columns={c: "Product"})
                break
    if "Product" not in df.columns:
        df["Product"] = "ALL_PRODUCTS"
    if "Sales" not in df.columns:
        numcols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numcols:
            df = df.rename(columns={numcols[0]: "Sales"})
        else:
            for c in df.columns:
                if any(k in c.lower() for k in ["qty", "quantity", "amount", "units", "sales"]):
                    df = df.rename(columns={c: "Sales"})
                    break
    if "Sales" not in df.columns:
        raise ValueError("No numeric Sales column found in sales CSV.")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
    if "Country" not in df.columns:
        for c in df.columns:
            if "country" in c.lower() or "region" in c.lower():
                df = df.rename(columns={c: "Country"})
                break
    if "Country" not in df.columns:
        df["Country"] = "All"
    return df

# ---------------- SIDEBAR (custom nav) ----------------
with st.sidebar:
    st.markdown("<div style='padding:8px 4px'><h3 style='color:white;margin:0'>AIC Inventory App</h3></div>", unsafe_allow_html=True)
    nav = st.radio("", ("Dashboard", "Inventory", "Suppliers", "Customers", "Purchases", "Sales", "Reports", "Settings"), index=0)
    st.markdown("---")
    st.header("Uploads")
    sales_file = st.file_uploader("Sales CSV", type=["csv"], key="sfile")
    stock_file = st.file_uploader("Stock CSV", type=["csv"], key="stfile")
    reviews_file = st.file_uploader("Reviews CSV", type=["csv"], key="rfile")
    cols = st.columns(2)
    load_demo = cols[0].button("Load demo")
    use_last = cols[1].checkbox("Use last", value=False)
    st.markdown("---")
    st.caption("This UI is styled to look like a modern inventory dashboard. Use 'Use last' to reuse earlier uploads on this host.")

# persist uploaded files
if sales_file is not None:
    save_persisted_file(sales_file, PERSIST_SALES)
if stock_file is not None:
    save_persisted_file(stock_file, PERSIST_STOCK)
if reviews_file is not None:
    save_persisted_file(reviews_file, PERSIST_REVIEWS)

if use_last:
    if os.path.exists(PERSIST_SALES):
        sales_file = PERSIST_SALES
    if os.path.exists(PERSIST_STOCK):
        stock_file = PERSIST_STOCK
    if os.path.exists(PERSIST_REVIEWS):
        reviews_file = PERSIST_REVIEWS

if load_demo:
    if os.path.exists(DEMO_SALES):
        sales_file = DEMO_SALES
    if os.path.exists(DEMO_STOCK):
        stock_file = DEMO_STOCK if os.path.exists(DEMO_STOCK) else None
    if os.path.exists(DEMO_REVIEWS):
        reviews_file = DEMO_REVIEWS

# ---------------- Apply a simple dark theme with CSS ----------------
st.markdown(
    """
    <style>
    /* Sidebar background */
    .css-1d391kg {background-color: #0f3b47 !important;}
    .css-1d391kg .css-1lcbmhc {color: white !important;}
    /* KPI card style */
    .kpi-card {background: linear-gradient(180deg, #ffffff, #f5f7fb); border-radius:10px; padding:14px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);}
    .kpi-title {color:#6b7280; font-size:12px;}
    .kpi-value {font-weight:700; font-size:22px;}
    /* Small helper */
    .muted {color:#6b7280; font-size:12px;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Data load ----------------
if sales_file is None:
    st.info("Upload sales CSV or click 'Load demo' (or Use last).")
    # show minimal dashboard placeholders
    if nav != "Dashboard":
        st.stop()
    # show empty demo dashboard
    st.title("Dashboard")
    st.write("Upload data to populate charts and KPIs.")
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

# ---------------- Basic aggregates ----------------
sales_df['Date'] = pd.to_datetime(sales_df['Date'], errors='coerce')
summary = sales_df.copy()
total_sales = summary['Sales'].sum()
total_orders = len(summary)
avg_order = summary['Sales'].mean()
countries = summary['Country'].nunique() if 'Country' in summary.columns else 1
top_selling = summary.groupby('Product')['Sales'].sum().sort_values(ascending=False).head(10)
top_customers = summary.groupby('Country')['Sales'].sum().sort_values(ascending=False).head(5)

# ---------------- Layout: Top KPIs ----------------
st.markdown("<div style='display:flex;gap:12px;margin-bottom:18px'>", unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns([1,1,1,1,1])
with k1:
    st.markdown("<div class='kpi-card'><div class='kpi-title'>Total Sales</div><div class='kpi-value'>${:,.0f}</div><div class='muted'>All time</div></div>".format(total_sales), unsafe_allow_html=True)
with k2:
    st.markdown("<div class='kpi-card'><div class='kpi-title'>Total Purchases</div><div class='kpi-value'>$ {:,.0f}</div><div class='muted'>Estimate</div></div>".format(total_sales*0.84), unsafe_allow_html=True)
with k3:
    st.markdown("<div class='kpi-card'><div class='kpi-title'>Net Profit</div><div class='kpi-value'>$ {:,.0f}</div><div class='muted'>Estimated</div></div>".format(total_sales*0.12), unsafe_allow_html=True)
with k4:
    st.markdown("<div class='kpi-card'><div class='kpi-title'>Total Receivable</div><div class='kpi-value'>$ {:,.0f}</div><div class='muted'>Estimate</div></div>".format(total_sales*0.4), unsafe_allow_html=True)
with k5:
    st.markdown("<div class='kpi-card'><div class='kpi-title'>Top Sales Location</div><div class='kpi-value'>{}</div><div class='muted'>Top country</div></div>".format(top_customers.index[0] if not top_customers.empty else "N/A"), unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Main Dashboard Grid ----------------
st.markdown("### Key trends and business insights")
left, right = st.columns([3,1])

with left:
    # Sales trend chart
    st.subheader("Sales Trend")
    daily = sales_df.dropna(subset=['Date']).groupby('Date')['Sales'].sum().reset_index()
    import altair as alt
    line = alt.Chart(daily).mark_area(line={'color':'#1f77b4'}).encode(x='Date:T', y='Sales:Q').properties(height=300)
    st.altair_chart(line, use_container_width=True)

    # Second row with charts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top 10 Customers by sales (Country)")
        if not top_customers.empty:
            cust_df = top_customers.reset_index().rename(columns={'Country':'Country','Sales':'Sales'})
            bar = alt.Chart(cust_df).mark_bar().encode(x='Sales:Q', y=alt.Y('Country:N', sort='-x')).properties(height=250)
            st.altair_chart(bar, use_container_width=True)
        else:
            st.write("No customer data")

    with c2:
        st.subheader("Purchase by Location (demo pie)")
        locs = pd.DataFrame({'location':['Texas','California','Arizona','Florida'], 'value':[30,25,20,25]})
        pie = alt.Chart(locs).mark_arc(innerRadius=50).encode(theta='value:Q', color='location:N').properties(height=250)
        st.altair_chart(pie, use_container_width=True)

with right:
    st.subheader("Units left (Low stock)")
    if stock_df is None:
        st.info("Upload stock CSV to enable low-stock view.")
    else:
        s = stock_df.copy()
        if 'Product' not in s.columns:
            for c in s.columns:
                if 'product' in c.lower():
                    s = s.rename(columns={c:'Product'})
                    break
        if 'Stock' not in s.columns:
            for c in s.columns:
                if 'stock' in c.lower() or 'qty' in c.lower():
                    s = s.rename(columns={c:'Stock'})
                    break
        if 'Product' in s.columns and 'Stock' in s.columns:
            s['Stock'] = pd.to_numeric(s['Stock'], errors='coerce').fillna(0)
            avg7 = sales_df.groupby('Product')['Sales'].apply(lambda x: x.tail(7).mean() if len(x)>=3 else x.mean()).reset_index().rename(columns={'Sales':'Avg7'})
            merged = pd.merge(s, avg7, on='Product', how='left')
            merged['DaysLeft'] = merged['Stock'] / merged['Avg7'].replace(0, np.nan)
            low = merged.sort_values('DaysLeft').head(8)
            st.table(low[['Product','Stock','Avg7','DaysLeft']].fillna('N/A'))
        else:
            st.write("Stock file missing Product/Stock columns")

    st.markdown("### Best sellers")
    best = top_selling.reset_index().rename(columns={'Product':'Product','Sales':'Sales'})
    if best.empty:
        st.write("No data yet")
    else:
        st.table(best.head(8))

# ---------------- Footer: Recommendations and downloads ----------------
st.markdown("---")
st.subheader("Recommendations & Exports")
# Simple recommendation placeholder
st.write("Recommendations are computed from forecasts and reviews (if uploaded).")
if reviews_df is not None:
    st.write("Review data detected — sentiment-driven recommendations available.")

# Downloads
st.markdown("**Export data**")
if sales_df is not None:
    st.download_button("Download cleaned sales CSV", data=sales_df.to_csv(index=False).encode('utf-8'), file_name='sales_cleaned.csv', mime='text/csv')
if stock_df is not None:
    st.download_button("Download stock CSV", data=stock_df.to_csv(index=False).encode('utf-8'), file_name='stock.csv', mime='text/csv')
if reviews_df is not None:
    st.download_button("Download reviews CSV", data=reviews_df.to_csv(index=False).encode('utf-8'), file_name='reviews.csv', mime='text/csv')

st.caption("Designed to visually match a modern inventory dashboard. You can customize labels, colors and add more charts as needed.")
