
"""
Business Inventory Management - Streamlit app (fixed)

Save this file as `business_inventory_dashboard_fixed.py` and run with:
    pip install -r requirements.txt
    streamlit run business_inventory_dashboard_fixed.py

Requirements (suggested):
    streamlit
    pandas
    numpy
    altair
    nltk
    prophet   # optional, for better forecasting (named 'prophet' or 'fbprophet' depending on installation)
    openai    # optional, only if you want LLM suggestions

This app:
 - accepts sales, stock, and reviews CSV files
 - falls back to simple average forecasting if Prophet is unavailable
 - computes VADER sentiment if reviews are provided
 - produces recommendations and allows CSV downloads
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import timedelta

st.set_page_config(page_title="Business Inventory Management", layout="wide")

st.markdown("# Business Inventory Management")
st.write("Combined dashboard: Sales forecasting, Inventory recommendations, and Review-driven insights.")

# ---------------- Sidebar ----------------
st.sidebar.header("Uploads & Settings")
sales_file = st.sidebar.file_uploader("Sales CSV", type=["csv"])
stock_file = st.sidebar.file_uploader("Stock CSV", type=["csv"])
reviews_file = st.sidebar.file_uploader("Reviews CSV (optional)", type=["csv"])
load_demo = st.sidebar.button("Load demo sales + stock + reviews")
use_llm = st.sidebar.checkbox("Enable LLM suggestions (OpenAI)", value=False)
openai_key = st.sidebar.text_input("OpenAI API key (sk-...)", type="password") if use_llm else None
safety_days = st.sidebar.number_input("Safety stock (days)", min_value=1, max_value=90, value=7)
forecast_horizon = st.sidebar.number_input("Forecast horizon (days)", min_value=7, max_value=180, value=30)

# Demo paths (only available if you place these demo CSVs in /mnt/data)
_demo_sales = "/mnt/data/demo_sales_dataset.csv"
_demo_stock = "/mnt/data/demo_stock_dataset.csv"
_demo_reviews = "/mnt/data/unstructured_reviews_demo_multilingual.csv"

if load_demo:
    # If demo files exist at the paths above, use their paths; streamlit can open local files by path in this runtime.
    sales_file = _demo_sales if os.path.exists(_demo_sales) else None
    stock_file = _demo_stock if os.path.exists(_demo_stock) else None
    reviews_file = _demo_reviews if os.path.exists(_demo_reviews) else None
    st.success("Demo file paths set (if files exist on the server).")

# ---------------- Helpers ----------------
@st.cache_data
def safe_read(path_or_buf):
    """Read a csv from an UploadedFile or a local path; try common encodings."""
    try:
        if hasattr(path_or_buf, "read"):  # UploadedFile
            path_or_buf.seek(0)
            df = pd.read_csv(path_or_buf)
        else:
            df = pd.read_csv(path_or_buf)
        return df, None
    except Exception as e:
        # try alternate encoding
        try:
            if hasattr(path_or_buf, "read"):
                path_or_buf.seek(0)
            df = pd.read_csv(path_or_buf, encoding="ISO-8859-1")
            return df, "ISO-8859-1"
        except Exception as e2:
            return None, str(e2)

def normalize_sales_df(df):
    """Ensure Date, Product, Sales, Country columns exist in the sales dataframe."""
    df = df.copy()
    # find a date-like column
    if "Date" not in df.columns:
        for c in df.columns:
            if "date" in c.lower():
                df = df.rename(columns={c: "Date"})
                break
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # product
    if "Product" not in df.columns:
        # try to find product-like column
        for c in df.columns:
            if "product" in c.lower() or "item" in c.lower():
                df = df.rename(columns={c: "Product"})
                break
    if "Product" not in df.columns:
        df["Product"] = "ALL_PRODUCTS"
    # sales numeric
    if "Sales" not in df.columns:
        numcols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numcols:
            df = df.rename(columns={numcols[0]: "Sales"})
        else:
            # try common names
            for c in df.columns:
                if any(k in c.lower() for k in ["qty", "quantity", "amount", "units", "sales"]):
                    df = df.rename(columns={c: "Sales"})
                    break
    # final coercion
    if "Sales" not in df.columns:
        raise ValueError("No numeric Sales column found in sales CSV.")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
    # country
    if "Country" not in df.columns:
        for c in df.columns:
            if "country" in c.lower() or "region" in c.lower():
                df = df.rename(columns={c: "Country"})
                break
    if "Country" not in df.columns:
        df["Country"] = "All"
    return df

# ---------------- Load sales ----------------
if sales_file is None:
    st.info("Upload sales CSV or click 'Load demo' to use demo data.")
    st.stop()

sales_df, _err = safe_read(sales_file)
if sales_df is None:
    st.error(f"Failed to read sales CSV: {_err}")
    st.stop()

try:
    sales_df = normalize_sales_df(sales_df)
except Exception as e:
    st.error(f"Error processing sales CSV: {e}")
    st.stop()

# ---------------- Load stock & reviews (optional) ----------------
stock_df = None
if stock_file is not None:
    stock_df, _ = safe_read(stock_file)
    if stock_df is None:
        st.warning("Failed to read stock CSV.")

reviews_df = None
if reviews_file is not None:
    reviews_df, _ = safe_read(reviews_file)
    if reviews_df is None:
        st.warning("Failed to read reviews CSV.")

# Aggregate daily per product-country, drop NaT dates
daily = sales_df.dropna(subset=["Date"]).groupby(["Date", "Country", "Product"], dropna=False)["Sales"].sum().reset_index()

# Sidebar filters
countries = ["All"] + sorted(daily["Country"].dropna().unique().tolist())
sel_country = st.sidebar.selectbox("Country", countries, index=0)
top_n = st.sidebar.slider("Top N products", min_value=1, max_value=50, value=10)
view = daily if sel_country == "All" else daily[daily["Country"] == sel_country].copy()
top_products = view.groupby("Product")["Sales"].sum().reset_index().sort_values("Sales", ascending=False).head(top_n)
if top_products.empty:
    st.warning("No products found for the selected country/time range.")
    st.stop()
prod_list = top_products["Product"].tolist()
sel_product = st.sidebar.selectbox("Select product", prod_list)

prod_ts = view[view["Product"] == sel_product].groupby("Date")["Sales"].sum().reset_index().sort_values("Date")
if prod_ts.empty:
    st.warning("No sales for selected product.")
    st.stop()

st.header(f"Insights — {sel_product} ({sel_country})")

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total sales", f"{int(prod_ts['Sales'].sum()):,}")
col2.metric("Avg daily", f"{prod_ts['Sales'].mean():.2f}")

cur_stock = None
if stock_df is not None:
    s = stock_df.copy()
    # try to normalize columns
    if "Product" not in s.columns:
        for c in s.columns:
            if "product" in c.lower():
                s = s.rename(columns={c: "Product"})
                break
    if "Stock" not in s.columns:
        for c in s.columns:
            if "stock" in c.lower() or "qty" in c.lower():
                s = s.rename(columns={c: "Stock"})
                break
    if "Product" in s.columns and "Stock" in s.columns:
        match = s[s["Product"].astype(str).str.lower() == str(sel_product).lower()]
        if not match.empty:
            try:
                cur_stock = int(pd.to_numeric(match["Stock"].iloc[0], errors="coerce"))
            except Exception:
                cur_stock = None

col3.metric("Current stock", cur_stock if cur_stock is not None else "N/A")

# ---------------- Forecast ----------------
use_prophet = False
try:
    from prophet import Prophet
    use_prophet = True
except Exception:
    use_prophet = False

forecast_df = None
if not use_prophet:
    st.info("Prophet not available — using recent-average forecast.")
    recent_avg = prod_ts['Sales'].tail(30).mean() if len(prod_ts) >= 7 else prod_ts['Sales'].mean()
    last_date = prod_ts['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=int(forecast_horizon), freq='D')
    fc = pd.DataFrame([{"ds": d, "yhat": recent_avg, "yhat_lower": recent_avg * 0.9, "yhat_upper": recent_avg * 1.1} for d in future_dates])
    forecast_df = fc
else:
    try:
        ts = prod_ts.rename(columns={"Date": "ds", "Sales": "y"})[["ds", "y"]]
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        m.fit(ts)
        future = m.make_future_dataframe(periods=int(forecast_horizon))
        fc = m.predict(future)
        forecast_df = fc
    except Exception as e:
        st.warning(f"Prophet prediction failed ({e}). Using simple fallback.")
        recent_avg = prod_ts['Sales'].tail(30).mean() if len(prod_ts) >= 7 else prod_ts['Sales'].mean()
        last_date = prod_ts['Date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=int(forecast_horizon), freq='D')
        fc = pd.DataFrame([{"ds": d, "yhat": recent_avg, "yhat_lower": recent_avg * 0.9, "yhat_upper": recent_avg * 1.1} for d in future_dates])
        forecast_df = fc

# ---------------- Sentiment (VADER) ----------------
sentiment_available = False
daily_sent = pd.DataFrame()
if reviews_df is not None:
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except Exception:
            nltk.download('vader_lexicon')
        sia = SentimentIntensityAnalyzer()
        # normalize review columns: try to find ReviewText, Rating, Product, Date
        r = reviews_df.copy()
        if "ReviewText" not in r.columns:
            for c in r.columns:
                if any(k in c.lower() for k in ["review", "text", "comment"]):
                    r = r.rename(columns={c: "ReviewText"})
                    break
        if "Product" not in r.columns:
            for c in r.columns:
                if "product" in c.lower() or "item" in c.lower():
                    r = r.rename(columns={c: "Product"})
                    break
        if "Date" not in r.columns:
            for c in r.columns:
                if "date" in c.lower():
                    r = r.rename(columns={c: "Date"})
                    break
        r["ReviewText"] = r["ReviewText"].astype(str)
        r["VADER_compound"] = r["ReviewText"].apply(lambda t: sia.polarity_scores(t)["compound"])
        r["Date"] = pd.to_datetime(r["Date"], errors="coerce")
        # ensure Product exists
        if "Product" not in r.columns:
            r["Product"] = "ALL_PRODUCTS"
        daily_sent = r.groupby(["Date", "Product"])["VADER_compound"].mean().reset_index().rename(columns={"VADER_compound": "Avg_Sentiment"})
        sentiment_available = True
        reviews_df = r  # keep normalized
    except Exception as e:
        st.warning(f"VADER sentiment failed or nltk not installed: {e}")
        sentiment_available = False

# ---------------- Time series plot ----------------
st.subheader("Sales & Sentiment time series")
import altair as alt
plot_df = prod_ts.copy()
chart_obs = alt.Chart(plot_df).mark_line().encode(x='Date:T', y='Sales:Q')

if sentiment_available:
    prod_sent = daily_sent[daily_sent["Product"] == sel_product].copy()
    merged = pd.merge(prod_ts, prod_sent, left_on="Date", right_on="Date", how="left")
    merged["Avg_Sentiment"] = merged["Avg_Sentiment"].fillna(method="ffill").fillna(0)
    base = alt.Chart(merged).encode(x='Date:T')
    sales_line = base.mark_line().encode(y='Sales:Q')
    sent_line = base.mark_line().encode(y=alt.Y('Avg_Sentiment:Q', title='Avg Sentiment'))
    st.altair_chart((sales_line + sent_line).properties(width=900, height=350), use_container_width=True)
else:
    st.altair_chart(chart_obs.properties(width=900, height=350), use_container_width=True)

# ---------------- Recommendations ----------------
st.subheader("Recommendations (review-aware)")
recs = []
for p in prod_list:
    ser = view[view["Product"] == p].groupby("Date")["Sales"].sum().reset_index().sort_values("Date")
    if ser.empty:
        continue
    recent_avg = ser["Sales"].tail(30).mean() if len(ser) >= 7 else ser["Sales"].mean()
    fsum = float(recent_avg * int(forecast_horizon))
    cur = None
    if stock_df is not None and "Product" in stock_df.columns:
        s_match = stock_df[stock_df["Product"].astype(str).str.lower() == str(p).lower()]
        if not s_match.empty and "Stock" in stock_df.columns:
            try:
                cur = float(pd.to_numeric(s_match["Stock"].iloc[0], errors="coerce"))
            except Exception:
                cur = None
    # review-driven adjustments
    sentiment_adj = 1.0
    defect_flag = False
    llm_action = None
    if reviews_df is not None:
        try:
            rprod = reviews_df[reviews_df["Product"].astype(str) == str(p)]
        except Exception:
            rprod = pd.DataFrame()
        if not rprod.empty:
            if "VADER_compound" in rprod.columns:
                avg_v = rprod["VADER_compound"].mean()
                if avg_v < -0.3:
                    sentiment_adj = 0.7
                elif avg_v < -0.1:
                    sentiment_adj = 0.9
                elif avg_v > 0.3:
                    sentiment_adj = 1.1
            else:
                if "Rating" in rprod.columns:
                    try:
                        avg_rating = pd.to_numeric(rprod["Rating"], errors="coerce").mean()
                        if avg_rating < 3.0:
                            sentiment_adj = 0.85
                    except Exception:
                        pass
            # defect detection via columns or keywords
            if "DefectLabel" in rprod.columns and rprod["DefectLabel"].notna().sum() > 0:
                defect_flag = True
            else:
                text_blob = " ".join(rprod["ReviewText"].astype(str).tolist()).lower() if "ReviewText" in rprod.columns else ""
                for kw in ["broken", "damaged", "defect", "miss", "overheat", "battery", "delay", "late", "scratch"]:
                    if kw in text_blob:
                        defect_flag = True
                        break
            # optional LLM action
            if use_llm and openai_key:
                try:
                    import openai as _openai
                    _openai.api_key = openai_key
                    prompt = "You are a concise product ops analyst. Given customer reviews, recommend one short action for inventory or product (examples: 'Halt restock','Investigate quality','Increase safety stock','No action'). Reviews:\\n\\n"
                    sample = "\\n".join(rprod["ReviewText"].astype(str).head(8).tolist())
                    prompt += sample
                    # best-effort: try ChatCompletion
                    resp = _openai.ChatCompletion.create(model='gpt-4o-mini', messages=[{'role': 'user', 'content': prompt}], max_tokens=80)
                    llm_action = resp['choices'][0]['message']['content'].strip().splitlines()[0]
                except Exception:
                    llm_action = None

    adjusted_fsum = fsum * sentiment_adj
    # Simple rule-based recommendation
    if cur is None or np.isnan(cur):
        recommendation = "No stock data"
    else:
        if defect_flag:
            recommendation = "Investigate & Hold (defects reported)"
        elif cur < adjusted_fsum * 0.9:
            recommendation = "Stock Up"
        elif cur > adjusted_fsum * 1.5:
            recommendation = "Reduce"
        else:
            recommendation = "Hold"
    recs.append({
        "Product": p,
        "ForecastDemand": round(fsum, 2),
        "AdjustedForecast": round(adjusted_fsum, 2),
        "CurrentStock": int(cur) if cur is not None and not np.isnan(cur) else None,
        "DefectFlag": defect_flag,
        "LLM_Action": llm_action,
        "Recommendation": recommendation
    })

recs_df = pd.DataFrame(recs)
st.dataframe(recs_df)

# ---------------- Download buttons ----------------
if reviews_df is not None:
    try:
        csv_bytes = reviews_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download enriched reviews CSV", data=csv_bytes, file_name="reviews_enriched.csv", mime="text/csv")
    except Exception:
        st.download_button("Download enriched reviews CSV", data=reviews_df.to_csv(index=False), file_name="reviews_enriched.csv")

try:
    recs_bytes = recs_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download recommendations CSV", data=recs_bytes, file_name="recommendations.csv", mime="text/csv")
except Exception:
    st.download_button("Download recommendations CSV", data=recs_df.to_csv(index=False), file_name="recommendations.csv")

st.markdown("---")
st.caption("Notes: Sentiment computed with NLTK VADER (no torch). LLM suggestions are optional and require an OpenAI API key. Defect flags influence recommendations conservatively: defects cause Hold/Investigate rather than blind stocking.")
