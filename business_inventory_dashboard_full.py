
"""
Business Inventory Dashboard - Full (navigation + preserved features + LLM)

Instructions:
 - For Streamlit Cloud: place this file as app.py in your GitHub repo, include a `data/` folder with demo files.
 - For local runs: the app can optionally persist last uploads to /mnt/data (ephemeral on cloud).
 - LLM: reads OPENAI_API_KEY from st.secrets["OPENAI_API_KEY"] if available, otherwise you may enter a key on the Settings page for the session.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import timedelta
from io import BytesIO
import matplotlib.pyplot as plt
import altair as alt

st.set_page_config(page_title="AIC Inventory - Full", layout="wide", initial_sidebar_state="expanded")

# Paths: for local/demo convenience. When deploying to Streamlit Cloud, include demo files under 'data/' in your repo.
BASE_DIR = os.path.dirname(__file__)
REPO_DATA_DIR = os.path.join(BASE_DIR, "data")
PERSIST_DIR = "/mnt/data"  # writable in some runtimes but ephemeral on Cloud
PERSIST_SALES = os.path.join(PERSIST_DIR, "last_sales.csv")
PERSIST_STOCK = os.path.join(PERSIST_DIR, "last_stock.csv")
PERSIST_REVIEWS = os.path.join(PERSIST_DIR, "last_reviews.csv")

DEMO_SALES = os.path.join(REPO_DATA_DIR, "demo_sales_dataset.csv")
DEMO_STOCK = os.path.join(REPO_DATA_DIR, "demo_stock_dataset.csv")
DEMO_REVIEWS = os.path.join(REPO_DATA_DIR, "demo_reviews_dataset.csv")
LOGO_PATH = os.path.join(REPO_DATA_DIR, "aic_logo.png")

# ----------------- Utility helpers -----------------
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

# ----------------- Sidebar (Navigation + Data) -----------------
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=120)
    st.title("AIC Inventory")
    nav = st.selectbox("Navigate", ["Dashboard", "Inventory", "Reports", "Settings"])
    st.markdown("---")
    st.header("Data: upload or demo")
    sales_upload = st.file_uploader("Sales CSV", type=["csv"], key="sales_up")
    stock_upload = st.file_uploader("Stock CSV", type=["csv"], key="stock_up")
    reviews_upload = st.file_uploader("Reviews CSV (optional)", type=["csv"], key="reviews_up")
    cols = st.columns(3)
    load_demo = cols[0].button("Load demo")
    use_last = cols[1].checkbox("Use last (local only)", value=False)
    restore_first = cols[2].button("Restore first upload")
    st.markdown("---")
    st.caption("Tip: On Streamlit Cloud include demo files under 'data/' in your repo and press Load demo. 'Use last' persists locally to /mnt/data if available.")

# If user uploaded, persist locally (useful for local dev; ephemeral on Cloud)
if sales_upload is not None:
    save_persisted_file(sales_upload, PERSIST_SALES)
if stock_upload is not None:
    save_persisted_file(stock_upload, PERSIST_STOCK)
if reviews_upload is not None:
    save_persisted_file(reviews_upload, PERSIST_REVIEWS)

# Restore first-uploaded behavior: we keep an optional "first_upload" snapshot in /mnt/data if present.
FIRST_SALES = os.path.join(PERSIST_DIR, "first_sales_snapshot.csv")
FIRST_STOCK = os.path.join(PERSIST_DIR, "first_stock_snapshot.csv")
FIRST_REVIEWS = os.path.join(PERSIST_DIR, "first_reviews_snapshot.csv")

def snapshot_if_missing(src_path, uploaded):
    # If uploaded this run and no snapshot exists, save it as "first" snapshot.
    try:
        if uploaded is not None and not os.path.exists(src_path):
            if hasattr(uploaded, "read"):
                uploaded.seek(0)
                with open(src_path, "wb") as f:
                    f.write(uploaded.read())
                # reset read pointer
                uploaded.seek(0)
    except Exception:
        pass

snapshot_if_missing(FIRST_SALES, sales_upload)
snapshot_if_missing(FIRST_STOCK, stock_upload)
snapshot_if_missing(FIRST_REVIEWS, reviews_upload)

# Determine which files to use (uploaded > use_last > demo in repo)
def choose_file(uploaded, persist_path, demo_path):
    # uploaded (UploadedFile or None)
    if uploaded is not None:
        return uploaded
    # use last persisted local
    if use_last and os.path.exists(persist_path):
        return persist_path
    # load demo from repo if requested
    if load_demo and os.path.exists(demo_path):
        return demo_path
    return None

sales_file = choose_file(sales_upload, PERSIST_SALES, DEMO_SALES)
stock_file = choose_file(stock_upload, PERSIST_STOCK, DEMO_STOCK)
reviews_file = choose_file(reviews_upload, PERSIST_REVIEWS, DEMO_REVIEWS)

# If user pressed restore_first, prefer the first snapshots (if they exist)
if restore_first:
    if os.path.exists(FIRST_SALES):
        sales_file = FIRST_SALES
    if os.path.exists(FIRST_STOCK):
        stock_file = FIRST_STOCK
    if os.path.exists(FIRST_REVIEWS):
        reviews_file = FIRST_REVIEWS

# ----------------- Load & validate sales -----------------
if sales_file is None:
    st.info("Upload Sales CSV, click Load demo (repo) or re-upload. The app requires sales data to run.")
    st.stop()

sales_df, err = safe_read(sales_file)
if sales_df is None:
    st.error(f"Failed to read sales CSV: {err}")
    st.stop()
try:
    sales_df = normalize_sales_df(sales_df)
except Exception as e:
    st.error(f"Error processing sales CSV: {e}")
    st.stop()

# Load optional files
stock_df = None
if stock_file is not None:
    stock_df, _ = safe_read(stock_file)
reviews_df = None
if reviews_file is not None:
    reviews_df, _ = safe_read(reviews_file)

# ----------------- LLM support (Settings page can set session key) -----------------
# Prefer st.secrets if present
OPENAI_KEY = None
if "OPENAI_API_KEY" in st.secrets:
    OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
# Allow settings page to set session-level key
if "session_openai_key" not in st.session_state:
    st.session_state["session_openai_key"] = None

# ----------------- Common aggregates & filters -----------------
sales_df['Date'] = pd.to_datetime(sales_df['Date'], errors='coerce')
min_date = sales_df['Date'].min()
max_date = sales_df['Date'].max()

# Page-specific content
if nav == "Dashboard":
    st.title("Dashboard — Key trends & insights")
    # Top KPIs
    total_sales = sales_df['Sales'].sum()
    total_orders = len(sales_df)
    avg_daily = sales_df.groupby('Date')['Sales'].sum().mean()

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total sales", f"${total_sales:,.0f}")
    k2.metric("Avg daily", f"{avg_daily:.2f}")
    k3.metric("Records", total_orders)
    k4.metric("Countries", sales_df['Country'].nunique() if 'Country' in sales_df.columns else 1)

    st.markdown("---")
    # Filters
    colf, colp = st.columns([3,1])
    with colf:
        date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    with colp:
        product_list = ["All"] + sorted(sales_df['Product'].unique().tolist())
        product_sel = st.selectbox("Product", product_list)

    # apply filters
    sdf = sales_df.copy()
    start, end = date_range
    sdf = sdf[(sdf['Date'] >= pd.to_datetime(start)) & (sdf['Date'] <= pd.to_datetime(end))]
    if product_sel != "All":
        sdf = sdf[sdf['Product'] == product_sel]

    # Charts layout
    left, right = st.columns([3,1])
    with left:
        st.subheader("Sales trend")
        daily = sdf.groupby('Date')['Sales'].sum().reset_index()
        chart = alt.Chart(daily).mark_area(opacity=0.7).encode(x='Date:T', y='Sales:Q').properties(height=320)
        st.altair_chart(chart, use_container_width=True)
        # Download PNG
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
        st.table(best.head(10))
        st.subheader("Units left (low stock)")
        if stock_df is None:
            st.info("Upload stock CSV or include demo_stock in 'data/' to view low-stock items.")
        else:
            sd = stock_df.copy()
            # normalize stock columns
            if 'Product' not in sd.columns:
                for c in sd.columns:
                    if 'product' in c.lower():
                        sd = sd.rename(columns={c:'Product'}); break
            if 'Stock' not in sd.columns:
                for c in sd.columns:
                    if 'stock' in c.lower() or 'qty' in c.lower():
                        sd = sd.rename(columns={c:'Stock'}); break
            if 'Product' in sd.columns and 'Stock' in sd.columns:
                sd['Stock'] = pd.to_numeric(sd['Stock'], errors='coerce').fillna(0)
                avg7 = sales_df.groupby('Product')['Sales'].apply(lambda x: x.tail(7).mean() if len(x)>=3 else x.mean()).reset_index().rename(columns={'Sales':'Avg7'})
                merged = pd.merge(sd, avg7, on='Product', how='left')
                merged['DaysLeft'] = merged['Stock'] / merged['Avg7'].replace(0, np.nan)
                st.table(merged.sort_values('DaysLeft').head(10)[['Product','Stock','Avg7','DaysLeft']].fillna('N/A'))
            else:
                st.write("Stock file present but missing Product/Stock columns.")

    st.markdown("---")
    st.subheader("Top customers by country")
    top_cust = sales_df.groupby('Country')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
    st.bar_chart(top_cust.set_index('Country').head(8))

elif nav == "Inventory":
    st.title("Inventory — Manage stock & view product details")
    # Show stock table; allow edits (download modified)
    if stock_df is None:
        st.info("Upload stock CSV or Load demo with demo_stock_dataset.csv in repo/data/")
    else:
        sd = stock_df.copy()
        st.subheader("Stock table")
        st.dataframe(sd)
        st.markdown("Adjust stock values (download, edit locally, and re-upload).")
        st.download_button("Download stock CSV", data=sd.to_csv(index=False).encode('utf-8'), file_name='stock_export.csv', mime='text/csv')

    st.markdown("---")
    st.subheader("Product demand preview (top products)")
    prod_summary = sales_df.groupby('Product')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
    st.table(prod_summary.head(20))

elif nav == "Reports":
    st.title("Reports — Forecasts, Sentiment & Recommendations")
    # Forecasting (Prophet optional)
    st.subheader("Forecasting (per product)")
    prod = st.selectbox("Select product for forecast", ["All"] + sorted(sales_df['Product'].unique().tolist()))
    horizon = st.number_input("Forecast horizon (days)", min_value=7, max_value=180, value=30)
    # build series
    if prod == "All":
        prod_ts = sales_df.groupby('Date')['Sales'].sum().reset_index().sort_values('Date')
    else:
        prod_ts = sales_df[sales_df['Product'] == prod].groupby('Date')['Sales'].sum().reset_index().sort_values('Date')
    if prod_ts.empty:
        st.warning("No time series data for selected product.")
    else:
        # try prophet
        use_prophet = False
        try:
            from prophet import Prophet
            use_prophet = True
        except Exception:
            use_prophet = False
        if use_prophet:
            ts = prod_ts.rename(columns={'Date':'ds','Sales':'y'})[['ds','y']]
            try:
                m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                m.fit(ts)
                future = m.make_future_dataframe(periods=horizon)
                fc = m.predict(future)
                fig = m.plot(fc)
                st.pyplot(fig)
                fc_out = fc[['ds','yhat','yhat_lower','yhat_upper']].tail(horizon)
                st.dataframe(fc_out)
                st.download_button("Download forecast CSV", data=fc_out.to_csv(index=False).encode('utf-8'), file_name='forecast.csv', mime='text/csv')
            except Exception as e:
                st.warning(f"Prophet error: {e}. Using simple fallback forecast.")
                use_prophet = False
        if not use_prophet:
            recent_avg = prod_ts['Sales'].tail(30).mean() if len(prod_ts) >= 7 else prod_ts['Sales'].mean()
            last_date = prod_ts['Date'].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=int(horizon), freq='D')
            fc = pd.DataFrame([{'ds': d, 'yhat': recent_avg, 'yhat_lower': recent_avg*0.9, 'yhat_upper': recent_avg*1.1} for d in future_dates])
            st.line_chart(fc.set_index('ds')['yhat'])
            st.dataframe(fc)
            st.download_button("Download simple forecast CSV", data=fc.to_csv(index=False).encode('utf-8'), file_name='forecast_simple.csv', mime='text/csv')

    # Sentiment + recommendations (if reviews provided)
    st.markdown("---")
    st.subheader("Customer reviews & recommendations")
    if reviews_df is None:
        st.info("Upload reviews CSV to enable sentiment-driven recommendations (or include demo_reviews_dataset.csv in repo/data).")
    else:
        # VADER sentiment
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            import nltk
            try:
                nltk.data.find('sentiment/vader_lexicon.zip')
            except Exception:
                nltk.download('vader_lexicon')
            sia = SentimentIntensityAnalyzer()
            r = reviews_df.copy()
            if 'ReviewText' not in r.columns:
                for c in r.columns:
                    if any(k in c.lower() for k in ['review','text','comment']):
                        r = r.rename(columns={c:'ReviewText'}); break
            r['ReviewText'] = r['ReviewText'].astype(str)
            r['VADER_compound'] = r['ReviewText'].apply(lambda t: sia.polarity_scores(t)['compound'])
            st.dataframe(r.head(50))
            # simple per-product sentiment and action
            agg = r.groupby('Product')['VADER_compound'].mean().reset_index().rename(columns={'VADER_compound':'AvgSentiment'})
            agg['Action'] = agg['AvgSentiment'].apply(lambda x: 'Investigate' if x < -0.2 else ('Monitor' if x < 0 else 'No action'))
            st.table(agg)
            st.download_button("Download enriched reviews", data=r.to_csv(index=False).encode('utf-8'), file_name='reviews_enriched.csv', mime='text/csv')
        except Exception as e:
            st.warning(f"Sentiment unavailable: {e}")

    # LLM-driven short recommendation (optional)
    st.markdown("---")
    st.subheader("LLM suggestion (optional)")
    chosen_product = st.selectbox("Choose product for LLM suggestion", ["None"] + sorted(sales_df['Product'].unique().tolist()))
    if chosen_product != "None":
        # determine key source: session state then st.secrets
        key = st.session_state.get('session_openai_key') or OPENAI_KEY
        use_llm = st.button("Get LLM suggestion")
        if use_llm:
            if not key:
                st.error("No OpenAI key available. Set it on the Settings page or in st.secrets as OPENAI_API_KEY.")
            else:
                try:
                    import openai
                    openai.api_key = key
                    sample_reviews = ""
                    if reviews_df is not None:
                        sample_reviews = "\\n".join(reviews_df[reviews_df['Product'] == chosen_product]['ReviewText'].astype(str).head(10).tolist())
                    prompt = f\"\"\"You are a concise product operations analyst. Given product name: {chosen_product} and customer reviews, recommend a single short inventory or product action (examples: 'Halt restock', 'Investigate quality', 'Increase safety stock', 'No action'). Reviews:\\n{sample_reviews}\"\"\"
                    resp = openai.ChatCompletion.create(model='gpt-4o-mini', messages=[{'role':'user','content':prompt}], max_tokens=120)
                    suggestion = resp['choices'][0]['message']['content'].strip()
                    st.success("LLM suggestion: " + suggestion)
                except Exception as e:
                    st.error(f"LLM call failed: {e}")

elif nav == "Settings":
    st.title("Settings")
    st.subheader("LLM / OpenAI configuration")
    st.write("Preferred: add OPENAI_API_KEY in Streamlit app secrets (st.secrets). Alternatively set a session key below for this session only.")
    if "OPENAI_API_KEY" in st.secrets:
        st.info("OPENAI_API_KEY found in st.secrets (will be used automatically).")
    key_input = st.text_input("Enter OpenAI API key for this session (will not persist to secrets)", type="password")
    if key_input:
        st.session_state['session_openai_key'] = key_input
        st.success("Session OpenAI key set for this browser session.")
    st.markdown("---")
    st.subheader("Misc")
    st.write("Use 'Restore first upload' in the sidebar to revert to the first uploaded snapshot (local only).")
    st.write("On Streamlit Cloud the filesystem is ephemeral; include demo data in your repo under the 'data/' folder.")

else:
    st.write("Unknown navigation selection.")
