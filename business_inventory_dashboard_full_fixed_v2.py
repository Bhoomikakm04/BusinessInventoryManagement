
"""
Inventory Management System - Full (v2)

Changes from previous:
 - App title changed to "Inventory Management System"
 - Navigation includes "Product Performance" which analyzes product reviews and provides improvement feedback.
 - "Free LLM" option: a built-in rule-based reviewer that provides suggestions without external APIs.
 - OpenAI LLM support kept as optional (uses st.secrets or session key).
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import timedelta
from io import BytesIO
import matplotlib.pyplot as plt
import altair as alt
import re

st.set_page_config(page_title="Inventory Management System", layout="wide", initial_sidebar_state="expanded")

# Paths and demo locations
BASE_DIR = os.path.dirname(__file__)
REPO_DATA_DIR = os.path.join(BASE_DIR, "data")
PERSIST_DIR = "/mnt/data"
PERSIST_SALES = os.path.join(PERSIST_DIR, "last_sales.csv")
PERSIST_STOCK = os.path.join(PERSIST_DIR, "last_stock.csv")
PERSIST_REVIEWS = os.path.join(PERSIST_DIR, "last_reviews.csv")

DEMO_SALES = os.path.join(REPO_DATA_DIR, "demo_sales_dataset.csv")
DEMO_STOCK = os.path.join(REPO_DATA_DIR, "demo_stock_dataset.csv")
DEMO_REVIEWS = os.path.join(REPO_DATA_DIR, "demo_reviews_dataset.csv")
LOGO_PATH = os.path.join(REPO_DATA_DIR, "aic_logo.png")  # you can replace this with your logo

# ---------------- Helpers ----------------
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
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
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
    df["Sales"] = pd.to_numeric(df["Sales"], errors='coerce').fillna(0)
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

# Free "LLM" rule-based analyzer for reviews (no external API)
def free_llm_review_suggester(reviews_series):
    """
    reviews_series: iterable of review text (strings)
    returns: list of suggestions and summary lines
    """
    texts = [str(t).lower() for t in reviews_series.astype(str).tolist() if str(t).strip()]
    if not texts:
        return ["No reviews available for this product."]

    # Basic sentiment proxy using word lists
    negative_keywords = ["bad","broken","damage","defect","dead","not working","slow","overheat","crash","issue","problem","delay","late","refund","return","disappoint","noisy","scratch"]
    positive_keywords = ["good","great","excellent","perfect","awesome","love","works","stable","fast","silent","recommend"]

    neg_count = sum(any(kw in t for kw in negative_keywords) for t in texts)
    pos_count = sum(any(kw in t for kw in positive_keywords) for t in texts)

    suggestions = []
    summary = []
    summary.append(f"Total reviews analyzed: {len(texts)}")
    summary.append(f"Positive-like reviews: {pos_count}, Negative-like reviews: {neg_count}")

    # Extract common complaint phrases via regex for noun phrases (simple)
    phrase_counter = {}
    for t in texts:
        # split sentences and find noun-like complaints: words surrounding 'broken', 'overheat', etc.
        for kw in negative_keywords:
            if kw in t:
                snippet = t[max(0, t.find(kw)-30): t.find(kw)+30]
                snippet = re.sub(r'\\s+', ' ', snippet).strip()
                phrase_counter[snippet] = phrase_counter.get(snippet, 0) + 1

    # Top complaint snippets
    top_complaints = sorted(phrase_counter.items(), key=lambda x: x[1], reverse=True)[:5]
    if top_complaints:
        summary.append("Frequent complaint snippets:")
        for s,c in top_complaints:
            summary.append(f"- ({c}) {s}")

    # Heuristics for suggestions
    if neg_count > pos_count:
        suggestions.append("Investigate quality issues: many negative reports detected.")
    if any("battery" in t for t in texts):
        suggestions.append("Check battery and power-related issues; consider improving instructions about battery handling.")
    if any("overheat" in t or "heat" in t for t in texts):
        suggestions.append("Review thermal design / cooling recommendations; provide clearer guidance on avoiding overheating.")
    if any("delay" in t or "late" in t for t in texts):
        suggestions.append("Investigate shipping/logistics partners — several complaints mention delays.")
    if any("compat" in t or "compatibility" in t for t in texts):
        suggestions.append("List compatible devices/firmware clearly; provide a compatibility matrix in product docs.")
    if any("noisy" in t for t in texts):
        suggestions.append("Address noise through hardware adjustments or note noise levels in product description.")

    # If few complaints and many positives:
    if neg_count == 0 and pos_count > 0:
        suggestions.append("Customer sentiment appears positive; consider promoting highlighted positive features in marketing.")

    if not suggestions:
        suggestions.append("No clear actionable improvements detected; consider collecting more detailed feedback or enabling a feedback form.")

    return summary + [""] + ["Suggested actions:"] + ["- " + s for s in suggestions]

# ---------------- Sidebar -----------------
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=110)
    st.title("Inventory Management System")
    nav = st.selectbox("Navigate", ["Dashboard", "Inventory", "Product Performance", "Reports", "Settings"])
    st.markdown("---")
    st.header("Data: upload or demo")
    sales_upload = st.file_uploader("Sales CSV", type=["csv"], key="sales_up2")
    stock_upload = st.file_uploader("Stock CSV", type=["csv"], key="stock_up2")
    reviews_upload = st.file_uploader("Reviews CSV (optional)", type=["csv"], key="reviews_up2")
    cols = st.columns(3)
    load_demo = cols[0].button("Load demo")
    use_last = cols[1].checkbox("Use last (local only)", value=False)
    restore_first = cols[2].button("Restore first upload")
    st.markdown("---")
    st.caption("Tip: For Streamlit Cloud include demo files under 'data/' in your repo and press Load demo.")

# Persist uploaded files locally (useful for local dev; ephemeral on Cloud)
if sales_upload is not None:
    save_persisted_file(sales_upload, PERSIST_SALES)
if stock_upload is not None:
    save_persisted_file(stock_upload, PERSIST_STOCK)
if reviews_upload is not None:
    save_persisted_file(reviews_upload, PERSIST_REVIEWS)

# Snapshot first uploads if missing
FIRST_SALES = os.path.join(PERSIST_DIR, "first_sales_snapshot.csv")
FIRST_STOCK = os.path.join(PERSIST_DIR, "first_stock_snapshot.csv")
FIRST_REVIEWS = os.path.join(PERSIST_DIR, "first_reviews_snapshot.csv")

def snapshot_if_missing(src_path, uploaded):
    try:
        if uploaded is not None and not os.path.exists(src_path):
            if hasattr(uploaded, "read"):
                uploaded.seek(0)
                with open(src_path, "wb") as f:
                    f.write(uploaded.read())
                uploaded.seek(0)
    except Exception:
        pass

snapshot_if_missing(FIRST_SALES, sales_upload)
snapshot_if_missing(FIRST_STOCK, stock_upload)
snapshot_if_missing(FIRST_REVIEWS, reviews_upload)

# Choose files: uploaded > use_last > demo
def choose_file(uploaded, persist_path, demo_path):
    if uploaded is not None:
        return uploaded
    if use_last and os.path.exists(persist_path):
        return persist_path
    if load_demo and os.path.exists(demo_path):
        return demo_path
    return None

sales_file = choose_file(sales_upload, PERSIST_SALES, DEMO_SALES)
stock_file = choose_file(stock_upload, PERSIST_STOCK, DEMO_STOCK)
reviews_file = choose_file(reviews_upload, PERSIST_REVIEWS, DEMO_REVIEWS)

if restore_first:
    if os.path.exists(FIRST_SALES):
        sales_file = FIRST_SALES
    if os.path.exists(FIRST_STOCK):
        stock_file = FIRST_STOCK
    if os.path.exists(FIRST_REVIEWS):
        reviews_file = FIRST_REVIEWS

# Load sales (required)
if sales_file is None:
    st.info("Upload Sales CSV, click Load demo (repo) or re-upload. Sales data is required to use the app.")
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

stock_df = None
if stock_file is not None:
    stock_df, _ = safe_read(stock_file)
reviews_df = None
if reviews_file is not None:
    reviews_df, _ = safe_read(reviews_file)

# LLM/OpenAI key handling
OPENAI_KEY = None
if "OPENAI_API_KEY" in st.secrets:
    OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
if "session_openai_key" not in st.session_state:
    st.session_state["session_openai_key"] = None

# Common preprocessing
sales_df['Date'] = pd.to_datetime(sales_df['Date'], errors='coerce')
min_date = sales_df['Date'].min()
max_date = sales_df['Date'].max()

# ---------------- Pages ----------------
if nav == "Dashboard":
    st.title("Dashboard — Key trends & insights")
    total_sales = sales_df['Sales'].sum()
    total_orders = len(sales_df)
    avg_daily = sales_df.groupby('Date')['Sales'].sum().mean()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total sales", f"${total_sales:,.0f}")
    c2.metric("Avg daily", f"{avg_daily:.2f}")
    c3.metric("Records", total_orders)
    c4.metric("Countries", sales_df['Country'].nunique() if 'Country' in sales_df.columns else 1)

    st.markdown("---")
    colf, colp = st.columns([3,1])
    with colf:
        date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    with colp:
        product_list = ["All"] + sorted(sales_df['Product'].unique().tolist())
        product_sel = st.selectbox("Product", product_list)

    sdf = sales_df.copy()
    start, end = date_range
    sdf = sdf[(sdf['Date'] >= pd.to_datetime(start)) & (sdf['Date'] <= pd.to_datetime(end))]
    if product_sel != "All":
        sdf = sdf[sdf['Product'] == product_sel]

    left, right = st.columns([3,1])
    with left:
        st.subheader("Sales trend")
        daily = sdf.groupby('Date')['Sales'].sum().reset_index()
        chart = alt.Chart(daily).mark_area(opacity=0.7).encode(x='Date:T', y='Sales:Q').properties(height=320)
        st.altair_chart(chart, use_container_width=True)
    with right:
        st.subheader("Best sellers")
        best = sdf.groupby('Product')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
        st.table(best.head(10))

elif nav == "Inventory":
    st.title("Inventory — Manage stock & view product details")
    if stock_df is None:
        st.info("Upload stock CSV or include demo_stock_dataset.csv in repo/data/")
    else:
        sd = stock_df.copy()
        st.subheader("Stock table")
        st.dataframe(sd)
        st.download_button("Download stock CSV", data=sd.to_csv(index=False).encode('utf-8'), file_name='stock_export.csv', mime='text/csv')

elif nav == "Product Performance":
    st.title("Product Performance — Reviews & Improvement Suggestions")
    product_list = ["All"] + sorted(sales_df['Product'].unique().tolist())
    prod = st.selectbox("Choose product", product_list)
    st.markdown("### Feedback source")
    col_a, col_b = st.columns(2)
    with col_a:
        use_free_llm = st.checkbox("Use free LLM (built-in rule-based suggestions)", value=True, help="No external API required")
    with col_b:
        use_openai_llm = st.checkbox("Use OpenAI (optional, requires API key in Settings or st.secrets)", value=False)

    # select reviews for product
    if reviews_df is None:
        st.info("Upload reviews CSV or include demo_reviews_dataset.csv in the repo/data folder to analyze reviews.")
    else:
        r = reviews_df.copy()
        if 'ReviewText' not in r.columns:
            for c in r.columns:
                if any(k in c.lower() for k in ['review','text','comment']):
                    r = r.rename(columns={c:'ReviewText'}); break
        if 'Product' not in r.columns:
            r['Product'] = 'ALL_PRODUCTS'
        r['ReviewText'] = r['ReviewText'].astype(str)
        if prod != "All":
            rprod = r[r['Product'] == prod].copy()
        else:
            rprod = r.copy()

        st.subheader("Sample reviews")
        st.write(rprod[['Date','Product','ReviewText']].head(30))

        st.markdown("---")
        st.subheader("Improvement suggestions")

        suggestions = []
        if use_free_llm:
            suggestions = free_llm_review_suggester(rprod['ReviewText'])
            for line in suggestions:
                st.write(line)
        elif use_openai_llm:
            # call OpenAI if key present
            key = st.session_state.get('session_openai_key') or OPENAI_KEY
            if not key:
                st.error("No OpenAI key available. Set in Settings or st.secrets as OPENAI_API_KEY.")
            else:
                try:
                    import openai
                    openai.api_key = key
                    sample_reviews = "\\n".join(rprod['ReviewText'].astype(str).head(30).tolist())
                    prompt = f"""You are a concise product operations analyst.
Given product name: {prod} and customer reviews, recommend a short list of specific improvements for the product and packaging, prioritized by importance. Reviews:\n{sample_reviews}"""
                    resp = openai.ChatCompletion.create(model='gpt-4o-mini', messages=[{'role':'user','content':prompt}], max_tokens=250)
                    suggestion = resp['choices'][0]['message']['content'].strip()
                    st.markdown(suggestion)
                except Exception as e:
                    st.error(f"OpenAI call failed: {e}")

elif nav == "Reports":
    st.title("Reports — Forecasts, Sentiment & Recommendations")
    st.subheader("Forecasting (per product)")
    prod = st.selectbox("Select product for forecast", ["All"] + sorted(sales_df['Product'].unique().tolist()))
    horizon = st.number_input("Forecast horizon (days)", min_value=7, max_value=180, value=30)
    if prod == "All":
        prod_ts = sales_df.groupby('Date')['Sales'].sum().reset_index().sort_values('Date')
    else:
        prod_ts = sales_df[sales_df['Product'] == prod].groupby('Date')['Sales'].sum().reset_index().sort_values('Date')
    if prod_ts.empty:
        st.warning("No time series data for selected product.")
    else:
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

    st.markdown("---")
    st.subheader("Customer reviews & sentiment (VADER)")
    if reviews_df is None:
        st.info("Upload reviews CSV to enable sentiment-driven recommendations (or include demo_reviews_dataset.csv in repo/data).")
    else:
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
            agg = r.groupby('Product')['VADER_compound'].mean().reset_index().rename(columns={'VADER_compound':'AvgSentiment'})
            agg['Action'] = agg['AvgSentiment'].apply(lambda x: 'Investigate' if x < -0.2 else ('Monitor' if x < 0 else 'No action'))
            st.table(agg)
            st.download_button("Download enriched reviews", data=r.to_csv(index=False).encode('utf-8'), file_name='reviews_enriched.csv', mime='text/csv')
        except Exception as e:
            st.warning(f"Sentiment unavailable: {e}")

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
