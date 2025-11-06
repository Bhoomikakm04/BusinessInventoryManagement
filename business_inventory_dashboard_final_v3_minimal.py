
import streamlit as st
import pandas as pd
import numpy as np
import os, json
from io import BytesIO
import matplotlib.pyplot as plt
import altair as alt

st.set_page_config(page_title="Inventory Management System", layout="wide")

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
PERSIST_DIR = "/mnt/data"
DEMO_SALES = os.path.join(DATA_DIR, "demo_sales_dataset.csv")
DEMO_STOCK = os.path.join(DATA_DIR, "demo_stock_dataset.csv")
DEMO_REVIEWS = os.path.join(DATA_DIR, "demo_reviews_dataset.csv")
LOGO_PATH = os.path.join(DATA_DIR, "aic_logo.png")

# helpers
@st.cache_data
def read_csv_safe(path_or_buf):
    try:
        if hasattr(path_or_buf, "read"):
            path_or_buf.seek(0)
            return pd.read_csv(path_or_buf)
        return pd.read_csv(path_or_buf)
    except Exception:
        return pd.read_csv(path_or_buf, encoding="ISO-8859-1")

def normalize_sales_df(df):
    df = df.copy()
    for c in df.columns:
        if "date" in c.lower():
            df = df.rename(columns={c: "Date"})
            break
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
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
                if any(k in c.lower() for k in ["qty","quantity","amount","units","sales"]):
                    df = df.rename(columns={c: "Sales"})
                    break
    if "Sales" not in df.columns:
        raise ValueError("No numeric sales column found")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
    if "Country" not in df.columns:
        for c in df.columns:
            if "country" in c.lower() or "region" in c.lower():
                df = df.rename(columns={c: "Country"})
                break
    if "Country" not in df.columns:
        df["Country"] = "All"
    return df

def persist_uploaded(uploaded, path):
    try:
        if uploaded is None: return False
        if hasattr(uploaded, "read"):
            uploaded.seek(0)
            with open(path, "wb") as f:
                f.write(uploaded.read())
            return True
        return False
    except Exception:
        return False

# Free rule-based LLM-like suggester
def free_review_suggester(reviews_series):
    texts = [str(t).lower() for t in reviews_series.astype(str).tolist() if str(t).strip()]
    if not texts:
        return ["No reviews available."]
    neg_kw = ["broken","defect","overheat","delay","refund","return","compat","slow","crash"]
    pos_kw = ["good","great","excellent","works","stable","fast","recommend"]
    neg_count = sum(any(k in t for k in neg_kw) for t in texts)
    pos_count = sum(any(k in t for k in pos_kw) for t in texts)
    out = [f"Analyzed {len(texts)} reviews — positives: {pos_count}, negatives: {neg_count}"]
    snippets = []
    for t in texts:
        for k in neg_kw:
            if k in t:
                start = max(0, t.find(k)-30)
                snippets.append(t[start:t.find(k)+30])
    snippets = snippets[:6]
    if snippets:
        out.append("Top complaint snippets:")
        for s in snippets:
            out.append("- " + s)
    if neg_count > pos_count:
        out.append("Action: Investigate quality issues and consider QC checks.")
    else:
        out.append("Action: No immediate quality issues detected. Monitor reviews.")
    return out

# Sidebar: only Load demo and file uploaders (no use_last or clear/remove buttons)
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=120)
    st.title("Inventory Management System")
    page = st.selectbox("Navigate", ["Dashboard", "Inventory", "Product Performance", "Reports"])
    st.markdown("---")
    st.header("Data")
    sales_up = st.file_uploader("Sales CSV", type=["csv"], key="sales_u")
    stock_up = st.file_uploader("Stock CSV", type=["csv"], key="stock_u")
    reviews_up = st.file_uploader("Reviews CSV (optional)", type=["csv"], key="reviews_u")
    load_demo = st.button("Load demo")  # only load demo button
    st.markdown("---")
    st.header("Forecast")
    forecast_horizon = st.number_input("Horizon (days)", min_value=7, max_value=180, value=30)
    safety_days = st.number_input("Safety days", min_value=1, max_value=90, value=7)
    st.markdown("---")
    st.caption("Tip: include demo files in data/ in the repo for Load demo to work on Streamlit Cloud.")

# persist uploads (still save last uploaded for convenience)
if sales_up is not None:
    persist_uploaded(sales_up, os.path.join(PERSIST_DIR, "last_sales.csv"))
if stock_up is not None:
    persist_uploaded(stock_up, os.path.join(PERSIST_DIR, "last_stock.csv"))
if reviews_up is not None:
    persist_uploaded(reviews_up, os.path.join(PERSIST_DIR, "last_reviews.csv"))

def choose_file(uploaded, demo_path, persist_name):
    # Preference: uploaded file > demo (if load_demo True and demo exists) > persisted last file (if present)
    if uploaded is not None:
        return uploaded
    if load_demo and os.path.exists(demo_path):
        return demo_path
    # keep persisted last as fallback (but user removed use_last UI; we keep as hidden fallback)
    persisted = os.path.join(PERSIST_DIR, persist_name)
    if os.path.exists(persisted):
        return persisted
    return None

sales_file = choose_file(sales_up, DEMO_SALES, "last_sales.csv")
stock_file = choose_file(stock_up, DEMO_STOCK, "last_stock.csv")
reviews_file = choose_file(reviews_up, DEMO_REVIEWS, "last_reviews.csv")

if sales_file is None:
    st.info("Please upload Sales CSV or include demo files in data/ and click Load demo.")
    st.stop()

# load data
try:
    sales_df = read_csv_safe(sales_file)
except Exception as e:
    st.error(f"Unable to read sales CSV: {e}")
    st.stop()
sales_df = normalize_sales_df(sales_df)
stock_df = None
if stock_file is not None:
    try:
        stock_df = read_csv_safe(stock_file)
    except Exception:
        stock_df = None
reviews_df = None
if reviews_file is not None:
    try:
        reviews_df = read_csv_safe(reviews_file)
    except Exception:
        reviews_df = None

sales_df['Date'] = pd.to_datetime(sales_df['Date'], errors='coerce')
min_date, max_date = sales_df['Date'].min(), sales_df['Date'].max()

# compute stock actions function (used both Inventory and Dashboard alerts)
def compute_stock_actions(sales_df, stock_df, reviews_df, horizon_days=30, safety_days=7):
    products = sorted(sales_df['Product'].unique().tolist())
    recs = []
    use_prophet = False
    try:
        from prophet import Prophet
        use_prophet = True
    except Exception:
        use_prophet = False
    for p in products:
        ts = sales_df[sales_df['Product']==p].groupby('Date')['Sales'].sum().reset_index().sort_values('Date')
        recent_avg = ts['Sales'].tail(30).mean() if len(ts)>=7 else (ts['Sales'].mean() if len(ts)>0 else 0.0)
        if use_prophet and not ts.empty:
            try:
                m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                dfp = ts.rename(columns={'Date':'ds','Sales':'y'})[['ds','y']]
                m.fit(dfp)
                future = m.make_future_dataframe(periods=horizon_days)
                fc = m.predict(future)
                last_date = ts['Date'].max()
                fc_future = fc[fc['ds'] > last_date].head(horizon_days)
                forecast = float(fc_future['yhat'].sum())
            except Exception:
                forecast = float(recent_avg * horizon_days)
        else:
            forecast = float(recent_avg * horizon_days)
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
            match = s[s['Product'].astype(str).str.lower()==str(p).lower()]
            if not match.empty:
                try:
                    cur_stock = float(pd.to_numeric(match['Stock'].iloc[0], errors='coerce'))
                except Exception:
                    cur_stock = None
        adj = 1.0
        if reviews_df is not None:
            r = reviews_df.copy()
            if 'VADER_compound' in r.columns:
                avg_v = r[r['Product']==p]['VADER_compound'].mean() if not r[r['Product']==p].empty else np.nan
                if not np.isnan(avg_v):
                    if avg_v < -0.2: adj = 0.85
                    elif avg_v > 0.3: adj = 1.15
        adjusted = forecast * adj
        safety_target = safety_days * (recent_avg if recent_avg>0 else 1)
        if cur_stock is None or np.isnan(cur_stock):
            action = "No stock data"
        else:
            if cur_stock < adjusted * 0.9 or cur_stock < safety_target:
                action = "Stock Up"
            elif cur_stock > adjusted * 1.5:
                action = "Reduce"
            else:
                action = "Hold"
        recs.append({"Product":p, "Action":action, "CurrentStock":cur_stock, "AdjustedForecast":adjusted, "RecentAvg":recent_avg})
    return pd.DataFrame(recs)

# Pages
if page == "Dashboard":
    st.title("Dashboard — Trends & Insights")
    col1, col2 = st.columns([3,1])
    with col1:
        date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    with col2:
        prod_options = ["All"] + sorted(sales_df['Product'].unique().tolist())
        prod_filter = st.selectbox("Product", prod_options)
    s, e = date_range
    sdf = sales_df[(sales_df['Date']>=pd.to_datetime(s)) & (sales_df['Date']<=pd.to_datetime(e))]
    if prod_filter != "All":
        sdf = sdf[sdf['Product']==prod_filter]
    total_sales = sdf['Sales'].sum()
    avg_daily = sdf.groupby('Date')['Sales'].sum().mean() if not sdf.empty else 0
    top_product = sdf.groupby('Product')['Sales'].sum().idxmax() if not sdf.empty else "N/A"
    c1, c2, c3 = st.columns(3)
    c1.metric("Total sales", f"{total_sales:,.0f}")
    c2.metric("Avg daily sales", f"{avg_daily:.2f}")
    c3.metric("Top product", top_product)
    st.markdown("---")
    left, right = st.columns([2,1])
    with left:
        st.subheader("Sales trend + MA7")
        daily = sdf.groupby('Date')['Sales'].sum().reset_index().sort_values('Date')
        if daily.empty:
            st.info("No sales in selected range.")
        else:
            daily['MA7'] = daily['Sales'].rolling(7, min_periods=1).mean()
            a = alt.Chart(daily).mark_area(opacity=0.4).encode(x='Date:T', y='Sales:Q')
            l = alt.Chart(daily).mark_line().encode(x='Date:T', y='Sales:Q')
            m = alt.Chart(daily).mark_line(strokeDash=[4,4]).encode(x='Date:T', y='MA7:Q')
            st.altair_chart((a + l + m).properties(height=360), use_container_width=True)
            buf = BytesIO()
            fig, ax = plt.subplots(figsize=(10,3))
            ax.plot(daily['Date'], daily['Sales'], label='Sales')
            ax.plot(daily['Date'], daily['MA7'], label='MA7', linestyle='--')
            ax.legend()
            fig.tight_layout()
            fig.savefig(buf, format='png')
            buf.seek(0)
            st.download_button("Download trend PNG", data=buf, file_name="sales_trend.png", mime="image/png")
    with right:
        st.subheader("Cumulative & Top products")
        if not daily.empty:
            cum = daily.copy(); cum['Cumulative'] = cum['Sales'].cumsum()
            st.line_chart(cum.set_index('Date')['Cumulative'])
        top10 = sdf.groupby('Product')['Sales'].sum().reset_index().sort_values('Sales', ascending=False).head(10)
        st.table(top10)
    st.markdown("---")
    st.subheader("Monthly pivot (Product x YYYY-MM)")
    heat = sales_df.copy(); heat['YM'] = heat['Date'].dt.to_period('M').astype(str)
    heatf = heat.groupby(['YM','Product'])['Sales'].sum().reset_index()
    if heatf.empty:
        st.write("No pivot data.")
    else:
        pivot = heatf.pivot(index='Product', columns='YM', values='Sales').fillna(0).astype(int)
        st.dataframe(pivot)

    # quick stock alerts summary
    st.markdown("---")
    st.subheader("Stock alerts (quick view)")
    alerts = compute_stock_actions(sales_df, stock_df, reviews_df, horizon_days=forecast_horizon, safety_days=safety_days)
    counts = alerts['Action'].value_counts().to_dict()
    st.info(f"Stock Up: {counts.get('Stock Up',0)}  |  Reduce: {counts.get('Reduce',0)}  |  Hold: {counts.get('Hold',0)}  |  No stock data: {counts.get('No stock data',0)}")
    if not alerts[alerts['Action']=='Stock Up'].empty:
        st.warning("Stock Up examples:"); st.write(alerts[alerts['Action']=='Stock Up'].head(8)[['Product','CurrentStock','AdjustedForecast']])
    if not alerts[alerts['Action']=='Reduce'].empty:
        st.success("Reduce examples:"); st.write(alerts[alerts['Action']=='Reduce'].head(8)[['Product','CurrentStock','AdjustedForecast']])

elif page == "Inventory":
    st.title("Inventory — Stock & Recommendations")
    if stock_df is None:
        st.info("Upload stock CSV to view stock table and recommendations.")
    else:
        st.subheader("Stock table"); st.dataframe(stock_df); st.download_button("Download stock CSV", data=stock_df.to_csv(index=False).encode('utf-8'), file_name="stock_export.csv", mime="text/csv")
    st.markdown("---")
    st.subheader("Stock action recommendations")
    prod_list = sorted(sales_df['Product'].unique().tolist())
    sel = st.multiselect("Products to evaluate", options=prod_list, default=prod_list[:8])
    recs = compute_stock_actions(sales_df, stock_df, reviews_df, horizon_days=forecast_horizon, safety_days=safety_days)
    if sel:
        recs = recs[recs['Product'].isin(sel)]
    st.dataframe(recs)
    st.download_button("Download stock recommendations CSV", data=recs.to_csv(index=False).encode('utf-8'), file_name="stock_recommendations.csv", mime="text/csv")

elif page == "Product Performance":
    st.title("Product Performance — Reviews & Suggestions")
    prod_list = ["All"] + sorted(sales_df['Product'].unique().tolist())
    prod = st.selectbox("Choose product", prod_list)
    use_free = st.checkbox("Use free LLM (rule-based)", value=True)
    #use_openai = st.checkbox("Use OpenAI (optional)", value=False)
    if reviews_df is None:
        st.info("Upload reviews CSV to analyze reviews.")
    else:
        r = reviews_df.copy()
        if 'ReviewText' not in r.columns:
            for c in r.columns:
                if any(k in c.lower() for k in ['review','text','comment']):
                    r = r.rename(columns={c:'ReviewText'}); break
        r['ReviewText'] = r['ReviewText'].astype(str)
        rprod = r if prod=="All" else r[r['Product']==prod]
        st.subheader("Sample reviews"); st.write(rprod[['Date','Product','ReviewText']].head(30))
        st.markdown("---"); st.subheader("Suggestions")
        if use_free:
            suggestions = free_review_suggester(rprod['ReviewText'])
            for s in suggestions: st.write(s)
        elif use_openai:
            key = st.session_state.get('session_openai_key') or st.secrets.get('OPENAI_API_KEY', None)
            if not key:
                st.error("OpenAI key not available. Set in Settings or st.secrets.")
            else:
                try:
                    import openai
                    openai.api_key = key
                    sample = "\\n".join(rprod['ReviewText'].astype(str).head(40).tolist())
                    prompt = f"""You are a concise product ops analyst. Given product: {prod} and customer reviews below, list 3 prioritized improvements. Reviews:\n{sample}"""
                    resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=250)
                    suggestion = resp['choices'][0]['message']['content'].strip()
                    st.markdown(suggestion)
                except Exception as e:
                    st.error(f"OpenAI call failed: {e}")

elif page == "Reports":
    st.title("Reports — Forecasting & Sentiment")
    prod = st.selectbox("Choose product", ["All"] + sorted(sales_df['Product'].unique().tolist()))
    if prod=="All":
        ts = sales_df.groupby('Date')['Sales'].sum().reset_index().sort_values('Date')
    else:
        ts = sales_df[sales_df['Product']==prod].groupby('Date')['Sales'].sum().reset_index().sort_values('Date')
    if ts.empty:
        st.warning("No series data to forecast.")
    else:
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
            recent_avg = ts['Sales'].tail(30).mean() if len(ts)>=7 else ts['Sales'].mean()
            future_dates = pd.date_range(start=ts['Date'].max()+pd.Timedelta(days=1), periods=forecast_horizon, freq='D')
            fc = pd.DataFrame([{'ds':d,'yhat':recent_avg} for d in future_dates])
            st.line_chart(fc.set_index('ds')['yhat'])
            st.dataframe(fc)
            st.info("Prophet not available — using simple average forecast.")
    st.markdown("---")
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



else:
    st.write("Unknown page selected.")
