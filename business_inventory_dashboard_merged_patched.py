
import streamlit as st
import pandas as pd, numpy as np, os, io
from datetime import timedelta
st.set_page_config(page_title="Business Inventory Management", layout="wide")

st.markdown("# Business Inventory Management")
st.write("Combined dashboard: Sales forecasting, Inventory recommendations, and Review-driven insights.")

# Sidebar uploads
st.sidebar.header("Uploads & Settings")
sales_file = st.sidebar.file_uploader("Sales CSV", type=["csv"])
stock_file = st.sidebar.file_uploader("Stock CSV", type=["csv"])
reviews_file = st.sidebar.file_uploader("Reviews CSV (optional)", type=["csv"])
load_demo = st.sidebar.button("Load demo sales + stock + reviews")
use_llm = st.sidebar.checkbox("Enable LLM suggestions (OpenAI)", value=False)
openai_key = st.sidebar.text_input("OpenAI API key (sk-...)", type="password") if use_llm else None
safety_days = st.sidebar.number_input("Safety stock (days)", min_value=1, max_value=90, value=7)
forecast_horizon = st.sidebar.number_input("Forecast horizon (days)", min_value=7, max_value=180, value=30)

# demo paths
_demo_sales = "/mnt/data/demo_sales_dataset.csv"
_demo_stock = "/mnt/data/demo_stock_dataset.csv"
_demo_reviews = "/mnt/data/unstructured_reviews_demo_multilingual.csv"

if load_demo:
    sales_file = _demo_sales if os.path.exists(_demo_sales) else None
    stock_file = _demo_stock if os.path.exists(_demo_stock) else None
    reviews_file = _demo_reviews if os.path.exists(_demo_reviews) else None
    st.success("Demo files loaded (if present)")

# Safe read CSV helper
def safe_read(path_or_buf):
    try:
        if hasattr(path_or_buf, "read"):
            df = pd.read_csv(path_or_buf)
        else:
            df = pd.read_csv(path_or_buf)
        return df, None
    except Exception as e:
        try:
            df = pd.read_csv(path_or_buf, encoding="ISO-8859-1")
            return df, "ISO-8859-1"
        except Exception as e2:
            return None, str(e2)

if sales_file is None:
    st.info("Upload sales CSV or click 'Load demo' to use demo data.")
    st.stop()
sales_df, _ = safe_read(sales_file)
if sales_df is None:
    st.error("Failed to read sales CSV.")
    st.stop()

# normalize sales columns
if "Date" not in sales_df.columns:
    # try to find date-like column
    for c in sales_df.columns:
        if "date" in c.lower():
            sales_df = sales_df.rename(columns={c:"Date"}); break
sales_df["Date"] = pd.to_datetime(sales_df["Date"], errors="coerce")
if "Product" not in sales_df.columns:
    sales_df["Product"] = "ALL_PRODUCTS"
if "Sales" not in sales_df.columns:
    # try numeric col
    numcols = sales_df.select_dtypes(include=[np.number]).columns.tolist()
    if numcols:
        sales_df = sales_df.rename(columns={numcols[0]:"Sales"})
    else:
        st.error("No Sales numeric column found."); st.stop()
sales_df["Sales"] = pd.to_numeric(sales_df["Sales"], errors="coerce").fillna(0)
if "Country" not in sales_df.columns:
    sales_df["Country"] = "All"

# read stock
stock_df = None
if stock_file is not None:
    stock_df, _ = safe_read(stock_file)

# read reviews
reviews_df = None
if reviews_file is not None:
    reviews_df, _ = safe_read(reviews_file)

# aggregate daily per product-country
daily = sales_df.groupby(["Date","Country","Product"])["Sales"].sum().reset_index()

# Sidebar filters
countries = ["All"] + sorted(daily["Country"].unique().tolist())
sel_country = st.sidebar.selectbox("Country", countries, index=0)
top_n = st.sidebar.slider("Top N products", min_value=1, max_value=50, value=10)
view = daily if sel_country=="All" else daily[daily["Country"]==sel_country].copy()
top_products = view.groupby("Product")["Sales"].sum().reset_index().sort_values("Sales", ascending=False).head(top_n)
prod_list = top_products["Product"].tolist()
sel_product = st.sidebar.selectbox("Select product", prod_list)

prod_ts = view[view["Product"]==sel_product].groupby("Date")["Sales"].sum().reset_index().sort_values("Date")
if prod_ts.empty:
    st.warning("No sales for selected product."); st.stop()

st.header(f"Insights — {sel_product} ({sel_country})")

# KPIs
col1,col2,col3 = st.columns(3)
col1.metric("Total sales", f"{int(prod_ts['Sales'].sum()):,}")
col2.metric("Avg daily", f"{prod_ts['Sales'].mean():.2f}")
cur_stock=None
if stock_df is not None:
    s = stock_df.copy()
    if "Product" not in s.columns:
        for c in s.columns:
            if "product" in c.lower(): s = s.rename(columns={c:"Product"})
    if "Stock" not in s.columns:
        for c in s.columns:
            if "stock" in c.lower(): s = s.rename(columns={c:"Stock"})
    match = s[s["Product"].astype(str).str.lower()==sel_product.lower()]
    if not match.empty and "Stock" in match.columns:
        cur_stock = int(pd.to_numeric(match["Stock"].iloc[0], errors="coerce"))
col3.metric("Current stock", cur_stock if cur_stock is not None else "N/A")

# Forecast (fallback simple average or Prophet if installed)
use_prophet = False
try:
    from prophet import Prophet
    use_prophet = True
except Exception:
    use_prophet = False

forecast_df = None
if not use_prophet:
    st.info("Prophet not available — using recent-average forecast.")
    recent_avg = prod_ts['Sales'].tail(30).mean() if len(prod_ts)>=7 else prod_ts['Sales'].mean()
    last_date = prod_ts['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')
    fc = pd.DataFrame([{"ds":d, "yhat": recent_avg, "yhat_lower":recent_avg*0.9, "yhat_upper":recent_avg*1.1} for d in future_dates])
    forecast_df = fc
else:
    try:
        ts = prod_ts.rename(columns={"Date":"ds","Sales":"y"})[["ds","y"]]
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        m.fit(ts)
        future = m.make_future_dataframe(periods=forecast_horizon)
        fc = m.predict(future)
        forecast_df = fc
    except Exception as e:
        st.warning(f"Prophet failed ({e}). Using recent-average fallback.")
        recent_avg = prod_ts['Sales'].tail(30).mean() if len(prod_ts)>=7 else prod_ts['Sales'].mean()
        last_date = prod_ts['Date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')
        fc = pd.DataFrame([{"ds":d, "yhat": recent_avg, "yhat_lower":recent_avg*0.9, "yhat_upper":recent_avg*1.1} for d in future_dates])
        forecast_df = fc

# Sentiment via VADER (no torch). Compute sentiment for reviews_df if available, aggregate daily per product
sentiment_available = False
if reviews_df is not None:
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except Exception:
            nltk.download('vader_lexicon')
        sia = SentimentIntensityAnalyzer()
        reviews_df["Text"] = reviews_df["ReviewText"].astype(str)
        reviews_df["VADER_compound"] = reviews_df["Text"].apply(lambda t: sia.polarity_scores(t)["compound"])
        # aggregate daily per product
        reviews_df["Date"] = pd.to_datetime(reviews_df["Date"], errors="coerce")
        daily_sent = reviews_df.groupby(["Date","Product"])["VADER_compound"].mean().reset_index().rename(columns={"VADER_compound":"Avg_Sentiment"})
        sentiment_available = True
    except Exception as e:
        st.warning(f"VADER sentiment failed or nltk not installed: {e}")
        sentiment_available = False

# Show time series and sentiment chart
st.subheader("Sales & Sentiment time series")
import altair as alt
plot_df = prod_ts.copy()
chart_obs = alt.Chart(plot_df).mark_line().encode(x='Date:T', y='Sales:Q')
if sentiment_available:
    # merge sentiment for this product
    prod_sent = daily_sent[daily_sent["Product"]==sel_product].copy()
    # align dates
    merged = pd.merge(prod_ts, prod_sent, left_on="Date", right_on="Date", how="left")
    merged["Avg_Sentiment"] = merged["Avg_Sentiment"].fillna(method="ffill").fillna(0)
    base = alt.Chart(merged).encode(x='Date:T')
    sales_line = base.mark_line(color='blue').encode(y='Sales:Q')
    sent_line = base.mark_line(color='orange').encode(y=alt.Y('Avg_Sentiment:Q', title='Avg Sentiment'))
    st.altair_chart((sales_line + sent_line).properties(width=900, height=350), use_container_width=True)
else:
    st.altair_chart(chart_obs.properties(width=900, height=350), use_container_width=True)

# Recommendations influenced by reviews/LLM
st.subheader("Recommendations (review-aware)")
# prepare top products recs
recs=[]
for p in prod_list:
    ser = view[view["Product"]==p].groupby("Date")["Sales"].sum().reset_index().sort_values("Date")
    if ser.empty: continue
    recent_avg = ser["Sales"].tail(30).mean() if len(ser)>=7 else ser["Sales"].mean()
    fsum = float(recent_avg * forecast_horizon)
    cur = None
    if stock_df is not None and "Product" in stock_df.columns:
        s_match = stock_df[stock_df["Product"].astype(str).str.lower()==p.lower()]
        if not s_match.empty and "Stock" in s_match.columns:
            cur = float(pd.to_numeric(s_match["Stock"].iloc[0], errors="coerce"))
    # review-driven adjustments
    sentiment_adj = 1.0
    defect_flag = False
    llm_action = None
    if reviews_df is not None:
        rprod = reviews_df[reviews_df["Product"]==p]
        if not rprod.empty:
            # average VADER if available else use rating-derived sentiment
            if "VADER_compound" in rprod.columns:
                avg_v = rprod["VADER_compound"].mean()
                # if strongly negative, reduce forecast demand by 10-30% (customers unhappy -> returns, less demand)
                if avg_v < -0.3:
                    sentiment_adj = 0.7
                elif avg_v < -0.1:
                    sentiment_adj = 0.9
                elif avg_v > 0.3:
                    sentiment_adj = 1.1
            else:
                avg_rating = rprod["Rating"].mean()
                if avg_rating < 3.0:
                    sentiment_adj = 0.85
            # detect defect labels (from DefectLabel column) or keyword heuristic
            if "DefectLabel" in rprod.columns and rprod["DefectLabel"].notna().sum()>0:
                defect_flag = True
            else:
                # keyword search in text
                text_blob = " ".join(rprod["ReviewText"].astype(str).tolist()).lower()
                for kw in ["broken","damaged","defect","miss","overheat","battery","delay","late","scratch"]:
                    if kw in text_blob:
                        defect_flag = True
                        break
            # if LLM enabled, try to get quick action item (if openai available)
            if use_llm and openai_key:
                try:
                    import openai as _openai
                    _openai.api_key = openai_key
                    prompt = "You are a product operations analyst. Given customer reviews, recommend one short action for inventory or product: e.g., 'Halt restock', 'Investigate quality', 'Increase safety stock', 'No action'. Reviews:\\n\\n"
                    sample = "\\n".join(rprod["ReviewText"].astype(str).head(8).tolist())
                    prompt += sample
                    resp = _client.chat.completions.create(model='gpt-4o-mini', messages=[{'role':'system','content':'You are concise.'},{'role':'user','content':prompt}], max_tokens=80)
                    llm_action = resp['choices'][0]['message']['content'].strip().splitlines()[0]
                except Exception as e:
                    llm_action = None
    adjusted_fsum = fsum * sentiment_adj
    # Simple rule-based recommendation
    if cur is None:
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
    recs.append({"Product":p, "ForecastDemand":round(fsum,2), "AdjustedForecast": round(adjusted_fsum,2), "CurrentStock": int(cur) if cur is not None and not np.isnan(cur) else None, "DefectFlag": defect_flag, "LLM_Action": llm_action, "Recommendation": recommendation})

recs_df = pd.DataFrame(recs)
st.dataframe(recs_df)

# Export enriched reviews and recs
if reviews_df is not None:
    st.download_button("Download enriched reviews CSV", data=reviews_df.to_csv(index=False), file_name="reviews_enriched.csv")
st.download_button("Download recommendations CSV", data=recs_df.to_csv(index=False), file_name="recommendations.csv")

st.markdown("---")
st.caption("Notes: Sentiment computed with NLTK VADER (no torch). LLM suggestions are optional and require an OpenAI API key. Defect flags influence recommendations conservatively: defects cause Hold/Investigate rather than blind stocking.")