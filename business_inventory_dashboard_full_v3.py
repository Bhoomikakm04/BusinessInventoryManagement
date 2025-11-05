
"""
Inventory Management System - Full v3

- Adds richer dashboard charts (moving average, cumulative sales, top products, monthly heatmap)
- Uses Prophet (if available) to forecast per-product demand over a horizon for stock recommendations;
  falls back to recent-average forecast if Prophet is missing.
- Produces explicit StockAction recommendations and downloadable CSV.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import timedelta
from io import BytesIO
import matplotlib.pyplot as plt
import altair as alt

st.set_page_config(page_title="Inventory Management System v3", layout="wide", initial_sidebar_state="expanded")

# Paths
BASE_DIR = os.path.dirname(__file__)
REPO_DATA_DIR = os.path.join(BASE_DIR, "data")
PERSIST_DIR = "/mnt/data"
PERSIST_SALES = os.path.join(PERSIST_DIR, "last_sales.csv")
PERSIST_STOCK = os.path.join(PERSIST_DIR, "last_stock.csv")
PERSIST_REVIEWS = os.path.join(PERSIST_DIR, "last_reviews.csv")

DEMO_SALES = os.path.join(REPO_DATA_DIR, "demo_sales_dataset.csv")
DEMO_STOCK = os.path.join(REPO_DATA_DIR, "demo_stock_dataset.csv")
DEMO_REVIEWS = os.path.join(REPO_DATA_DIR, "demo_reviews_dataset.csv")
LOGO_PATH = os.path.join(REPO_DATA_DIR, "aic_logo.png")

# Helpers
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

# Sidebar
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=110)
    st.title("Inventory Management System v3")
    nav = st.selectbox("Navigate", ["Dashboard", "Inventory", "Product Performance", "Reports", "Settings"])
    st.markdown("---")
    st.header("Data")
    sales_upload = st.file_uploader("Sales CSV", type=["csv"], key="sales_v3")
    stock_upload = st.file_uploader("Stock CSV", type=["csv"], key="stock_v3")
    reviews_upload = st.file_uploader("Reviews CSV (optional)", type=["csv"], key="reviews_v3")
    cols = st.columns(3)
    load_demo = cols[0].button("Load demo")
    use_last = cols[1].checkbox("Use last (local only)", value=False)
    restore_first = cols[2].button("Restore first upload")
    st.markdown("---")
    st.header("Forecast & Recs")
    forecast_horizon = st.number_input("Forecast horizon (days)", min_value=7, max_value=180, value=30)
    safety_days = st.number_input("Safety stock (days)", min_value=1, max_value=90, value=7)
    st.markdown("---")
    st.caption("Tip: On Streamlit Cloud include demo files in 'data/' in the repo and press Load demo.")

# persist uploads locally
if sales_upload is not None:
    save_persisted_file(sales_upload, PERSIST_SALES)
if stock_upload is not None:
    save_persisted_file(stock_upload, PERSIST_STOCK)
if reviews_upload is not None:
    save_persisted_file(reviews_upload, PERSIST_REVIEWS)

# choose files
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

# restore first snapshots if requested
FIRST_SALES = os.path.join(PERSIST_DIR, "first_sales_snapshot.csv")
if restore_first and os.path.exists(FIRST_SALES):
    sales_file = FIRST_SALES

# load mandatory sales
if sales_file is None:
    st.info("Upload Sales CSV, click Load demo (repo) or re-upload. Sales data required.")
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

# basic preprocessing
sales_df['Date'] = pd.to_datetime(sales_df['Date'], errors='coerce')
min_date, max_date = sales_df['Date'].min(), sales_df['Date'].max()

# Dashboard with extra charts
if nav == "Dashboard":
    st.title("Dashboard — Trends & Insights")
    # Filters
    colf, colp = st.columns([3,1])
    with colf:
        date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    with colp:
        prod_options = ["All"] + sorted(sales_df['Product'].unique().tolist())
        prod_filter = st.selectbox("Product", prod_options)

    sdf = sales_df.copy()
    start, end = date_range
    sdf = sdf[(sdf['Date'] >= pd.to_datetime(start)) & (sdf['Date'] <= pd.to_datetime(end))]
    if prod_filter != "All":
        sdf = sdf[sdf['Product'] == prod_filter]

    # KPI row
    total_sales = sdf['Sales'].sum()
    avg_daily = sdf.groupby('Date')['Sales'].sum().mean() if not sdf.empty else 0
    top_product = sdf.groupby('Product')['Sales'].sum().idxmax() if not sdf.empty else "N/A"
    k1,k2,k3 = st.columns(3)
    k1.metric("Total sales", f"${total_sales:,.0f}")
    k2.metric("Avg daily sales", f"{avg_daily:.2f}")
    k3.metric("Top product", top_product)

    st.markdown("---")
    # charts grid
    c1, c2 = st.columns([2,1])
    with c1:
        st.subheader("Sales trend (area) + Moving average")
        daily = sdf.groupby('Date')['Sales'].sum().reset_index().sort_values('Date')
        if daily.empty:
            st.info("No sales in selected range.")
        else:
            daily['MA_7'] = daily['Sales'].rolling(window=7, min_periods=1).mean()
            chart_area = alt.Chart(daily).mark_area(opacity=0.5).encode(x='Date:T', y='Sales:Q')
            chart_line = alt.Chart(daily).mark_line(color='steelblue').encode(x='Date:T', y='Sales:Q')
            chart_ma = alt.Chart(daily).mark_line(strokeDash=[4,4], color='orange').encode(x='Date:T', y='MA_7:Q')
            st.altair_chart((chart_area + chart_line + chart_ma).properties(height=360), use_container_width=True)

            # export PNG for trend
            buf = BytesIO()
            fig, ax = plt.subplots(figsize=(10,3))
            ax.plot(daily['Date'], daily['Sales'], label='Sales')
            ax.plot(daily['Date'], daily['MA_7'], label='MA 7', linestyle='--')
            ax.set_title('Sales trend + MA7')
            ax.legend()
            fig.tight_layout()
            fig.savefig(buf, format='png')
            buf.seek(0)
            st.download_button("Download trend PNG", data=buf, file_name="sales_trend_ma.png', mime='image/png')

    with c2:
        st.subheader("Cumulative sales & Top products")
        cum = daily.copy()
        cum['Cumulative'] = cum['Sales'].cumsum()
        st.line_chart(cum.set_index('Date')['Cumulative'])
        top10 = sdf.groupby('Product')['Sales'].sum().reset_index().sort_values('Sales', ascending=False).head(10)
        st.table(top10)

    st.markdown("---")
    st.subheader("Monthly sales heatmap (by product)")
    # prepare heatmap-like table: month vs product sales sum
    heat = sales_df.copy()
    heat['YearMonth'] = heat['Date'].dt.to_period('M').astype(str)
    heatf = heat.groupby(['YearMonth','Product'])['Sales'].sum().reset_index()
    if heatf.empty:
        st.write("No data for heatmap.")
    else:
        # pivot for display
        pivot = heatf.pivot(index='Product', columns='YearMonth', values='Sales').fillna(0)
        st.dataframe(pivot.astype(int))

# Inventory page
elif nav == "Inventory":
    st.title("Inventory — Stock overview & recommendations")
    if stock_df is None:
        st.info("Upload stock CSV or include demo_stock in repo/data to enable stock view.")
    else:
        sd = stock_df.copy()
        st.subheader("Stock table")
        st.dataframe(sd)
        st.download_button("Download stock CSV", data=sd.to_csv(index=False).encode('utf-8'), file_name='stock_export.csv', mime='text/csv')

    # Stock action recommendations (per product) using Prophet if available else recent-average
    st.markdown("---")
    st.subheader("Stock action recommendations")
    prod_list = sorted(sales_df['Product'].unique().tolist())
    # allow multi-select or all
    sel_prods = st.multiselect("Products to evaluate", options=prod_list, default=prod_list[:10])

    # try prophet
    use_prophet = False
    try:
        from prophet import Prophet
        use_prophet = True
    except Exception:
        use_prophet = False

    rec_rows = []
    for p in sel_prods:
        ts = sales_df[sales_df['Product'] == p].groupby('Date')['Sales'].sum().reset_index().sort_values('Date')
        if ts.empty:
            recent_avg = 0.0
        else:
            recent_avg = ts['Sales'].tail(30).mean() if len(ts) >= 7 else ts['Sales'].mean()

        # forecast demand over forecast_horizon
        if use_prophet and not ts.empty:
            try:
                m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                dfp = ts.rename(columns={'Date':'ds','Sales':'y'})[['ds','y']]
                m.fit(dfp)
                future = m.make_future_dataframe(periods=int(forecast_horizon))
                fc = m.predict(future)
                # sum predictions in the future horizon only
                last_date = ts['Date'].max()
                fc_future = fc[fc['ds'] > last_date].head(int(forecast_horizon))
                forecast_demand = float(fc_future['yhat'].sum())
            except Exception:
                forecast_demand = float(recent_avg * int(forecast_horizon))
        else:
            forecast_demand = float(recent_avg * int(forecast_horizon))

        # current stock lookup
        cur_stock = None
        if stock_df is not None:
            s = stock_df.copy()
            if 'Product' not in s.columns:
                for c in s.columns:
                    if 'product' in c.lower():
                        s = s.rename(columns={c:'Product'}); break
            if 'Stock' not in s.columns:
                for c in s.columns:
                    if 'stock' in c.lower() or 'qty' in c.lower():
                        s = s.rename(columns={c:'Stock'}); break
            match = s[s['Product'].astype(str).str.lower() == str(p).lower()]
            if not match.empty:
                try:
                    cur_stock = float(pd.to_numeric(match['Stock'].iloc[0], errors='coerce'))
                except Exception:
                    cur_stock = None

        # sentiment adjustment (simple): if reviews present use avg rating or vader if available
        adj = 1.0
        if reviews_df is not None:
            r = reviews_df.copy()
            if 'VADER_compound' in r.columns:
                avg_v = r[r['Product'] == p]['VADER_compound'].mean() if not r[r['Product'] == p].empty else np.nan
                if not np.isnan(avg_v):
                    if avg_v < -0.2: adj = 0.8
                    elif avg_v > 0.3: adj = 1.15
            elif 'Rating' in r.columns:
                avg_r = pd.to_numeric(r[r['Product'] == p]['Rating'], errors='coerce').mean() if not r[r['Product'] == p].empty else np.nan
                if not np.isnan(avg_r):
                    if avg_r < 3: adj = 0.85
                    elif avg_r >=4: adj = 1.1

        adjusted_forecast = forecast_demand * adj

        # decision rules
        if cur_stock is None or np.isnan(cur_stock):
            action = "No stock data"
        else:
            # incorporate safety stock days
            safety_target = safety_days * (recent_avg if recent_avg>0 else 1)
            if cur_stock < adjusted_forecast * 0.9 or cur_stock < safety_target:
                action = "Stock Up"
            elif cur_stock > adjusted_forecast * 1.5:
                action = "Reduce"
            else:
                action = "Hold"

        days_left = None
        if recent_avg > 0 and cur_stock is not None and not np.isnan(cur_stock):
            days_left = cur_stock / recent_avg

        rec_rows.append({
            "Product": p,
            "RecentAvgDaily": round(float(recent_avg),2),
            f"Forecast_{forecast_horizon}d": round(float(forecast_demand),2),
            "AdjFactor": round(float(adj),2),
            "AdjustedForecast": round(float(adjusted_forecast),2),
            "CurrentStock": int(cur_stock) if cur_stock is not None and not np.isnan(cur_stock) else None,
            "DaysLeft": round(float(days_left),1) if days_left is not None else None,
            "Action": action
        })

    rec_df = pd.DataFrame(rec_rows).sort_values(by="Action", key=lambda s: s.map({'Stock Up':0,'No stock data':1,'Hold':2,'Reduce':3}))
    st.dataframe(rec_df)
    st.download_button("Download stock recommendations CSV", data=rec_df.to_csv(index=False).encode('utf-8'), file_name='stock_recommendations.csv', mime='text/csv')

# Product Performance page
elif nav == "Product Performance":
    st.title("Product Performance — Reviews & Improvement Suggestions")
    product_list = ["All"] + sorted(sales_df['Product'].unique().tolist())
    prod = st.selectbox("Choose product", product_list)
    st.markdown("### Feedback source")
    col_a, col_b = st.columns(2)
    with col_a:
        use_free_llm = st.checkbox("Use free LLM (built-in rule-based suggestions)", value=True)
    with col_b:
        use_openai_llm = st.checkbox("Use OpenAI (optional)", value=False)

    if reviews_df is None:
        st.info("Upload reviews CSV or include demo_reviews in repo/data to analyze reviews.")
    else:
        r = reviews_df.copy()
        if 'ReviewText' not in r.columns:
            for c in r.columns:
                if any(k in c.lower() for k in ['review','text','comment']):
                    r = r.rename(columns={c:'ReviewText'}); break
        r['ReviewText'] = r['ReviewText'].astype(str)
        if prod != "All":
            rprod = r[r['Product'] == prod].copy()
        else:
            rprod = r.copy()
        st.subheader("Sample reviews")
        st.write(rprod[['Date','Product','ReviewText']].head(30))
        st.markdown("---")
        st.subheader("Suggested improvements")
        if use_free_llm:
            texts = [str(t).lower() for t in rprod['ReviewText'].astype(str).tolist() if str(t).strip()]
            if not texts:
                st.write("No reviews available.")
            else:
                complaints = [t for t in texts if any(w in t for w in ['broken','defect','overheat','delay','late','refund','return','compat'])]
                positives = [t for t in texts if any(w in t for w in ['good','great','excellent','works','stable','fast'])]
                st.write(f"Analyzed {len(texts)} reviews — {len(complaints)} complaint-like, {len(positives)} praise-like.")
                if complaints:
                    st.write("Top complaint examples:")
                    for ex in complaints[:6]:
                        st.write("-", ex[:200])
                    st.write("Actions:")
                    if any('overheat' in c or 'heat' in c for c in complaints):
                        st.write("- Investigate thermal/ cooling issues; add cooling recommendations in product docs.")
                    if any('broken' in c or 'defect' in c for c in complaints):
                        st.write("- Audit QC for affected batches; consider temporary hold on restock until root cause identified.")
                    if any('compat' in c for c in complaints):
                        st.write("- Publish compatibility matrix and firmware guidance.")
                else:
                    st.write("No frequent complaints detected. Consider highlighting positive features in marketing.")
        elif use_openai_llm:
            key = st.session_state.get('session_openai_key') or st.secrets.get('OPENAI_API_KEY', None)
            if not key:
                st.error("OpenAI key not available. Set it in Settings or st.secrets as OPENAI_API_KEY.")
            else:
                try:
                    import openai
                    openai.api_key = key
                    sample_reviews = "\\n".join(rprod['ReviewText'].astype(str).head(40).tolist())
                    prompt = f\"\"\"You are a concise product ops analyst. Given product: {prod} and customer reviews below, list prioritized improvements to product and packaging (3-6 short bullets). Reviews:\\n{sample_reviews}\"\"\"
                    resp = openai.ChatCompletion.create(model='gpt-4o-mini', messages=[{'role':'user','content':prompt}], max_tokens=250)
                    suggestion = resp['choices'][0]['message']['content'].strip()
                    st.markdown(suggestion)
                except Exception as e:
                    st.error(f"OpenAI call failed: {e}")

elif nav == "Reports":
    st.title("Reports — Forecasting & Sentiment")
    st.subheader("Forecasting examples")
    prod = st.selectbox("Choose product for example forecast", ["All"] + sorted(sales_df['Product'].unique().tolist()))
    if prod == "All":
        ts = sales_df.groupby('Date')['Sales'].sum().reset_index().sort_values('Date')
    else:
        ts = sales_df[sales_df['Product'] == prod].groupby('Date')['Sales'].sum().reset_index().sort_values('Date')
    if ts.empty:
        st.warning("No series data to forecast.")
    else:
        # try prophet for forecasting plot
        try:
            from prophet import Prophet
            use_prophet = True
        except Exception:
            use_prophet = False
        if use_prophet:
            dfp = ts.rename(columns={'Date':'ds','Sales':'y'})[['ds','y']]
            m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
            m.fit(dfp)
            future = m.make_future_dataframe(periods=forecast_horizon)
            fc = m.predict(future)
            fig = m.plot(fc)
            st.pyplot(fig)
            st.dataframe(fc[['ds','yhat','yhat_lower','yhat_upper']].tail(forecast_horizon))
        else:
            recent_avg = ts['Sales'].tail(30).mean() if len(ts) >= 7 else ts['Sales'].mean()
            future_dates = pd.date_range(start=ts['Date'].max() + pd.Timedelta(days=1), periods=int(forecast_horizon), freq='D')
            fc = pd.DataFrame([{'ds': d, 'yhat': recent_avg} for d in future_dates])
            st.line_chart(fc.set_index('ds')['yhat'])
            st.dataframe(fc)
            st.info("Prophet not available — using simple average forecast.")

    st.markdown("---")
    st.subheader("Sentiment (VADER) if reviews uploaded")
    if reviews_df is None:
        st.info("Upload reviews CSV to enable sentiment analysis.")
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
        except Exception as e:
            st.warning(f"Sentiment analysis unavailable: {e}")

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
    st.write("On Streamlit Cloud the filesystem is ephemeral; include demo data in your repo under the 'data/' folder.")

else:
    st.write("Unknown navigation selection.")
