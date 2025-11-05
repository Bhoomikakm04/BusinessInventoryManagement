
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import timedelta
from io import BytesIO
import matplotlib.pyplot as plt
import altair as alt

st.set_page_config(page_title="Inventory Management System v3 (Fixed)", layout="wide")

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
PERSIST_DIR = "/mnt/data"
DEMO_SALES = os.path.join(DATA_DIR, "demo_sales_dataset.csv")
DEMO_STOCK = os.path.join(DATA_DIR, "demo_stock_dataset.csv")
DEMO_REVIEWS = os.path.join(DATA_DIR, "demo_reviews_dataset.csv")
LOGO_PATH = os.path.join(DATA_DIR, "aic_logo.png")

@st.cache_data
def safe_read(path_or_buf):
    try:
        if hasattr(path_or_buf, "read"):
            path_or_buf.seek(0)
            return pd.read_csv(path_or_buf), None
        return pd.read_csv(path_or_buf), None
    except Exception as e:
        try:
            if hasattr(path_or_buf, "read"):
                path_or_buf.seek(0)
            return pd.read_csv(path_or_buf, encoding="ISO-8859-1"), "ISO-8859-1"
        except Exception as e2:
            return None, str(e2)

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
        raise ValueError("No numeric Sales column found.")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
    for c in df.columns:
        if "country" in c.lower() or "region" in c.lower():
            df = df.rename(columns={c: "Country"})
            break
    if "Country" not in df.columns:
        df["Country"] = "All"
    return df

def save_file(uploaded, path):
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

# Sidebar UI
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=100)
    st.title("Inventory Management System")
    page = st.selectbox("Navigate", ["Dashboard", "Inventory", "Product Performance", "Reports", "Settings"])
    st.markdown("---")
    st.header("Data")
    sales_up = st.file_uploader("Sales CSV", type=["csv"])
    stock_up = st.file_uploader("Stock CSV", type=["csv"])
    reviews_up = st.file_uploader("Reviews CSV (optional)", type=["csv"])
    cols = st.columns(3)
    load_demo = cols[0].button("Load demo")
    use_last = cols[1].checkbox("Use last (local only)")
    st.markdown("---")
    st.caption("Deploy to Streamlit Cloud: include demo files in the repo 'data/' folder.")

# Persist uploaded files locally for convenience
if sales_up is not None:
    save_file(sales_up, os.path.join(PERSIST_DIR, "last_sales.csv"))
if stock_up is not None:
    save_file(stock_up, os.path.join(PERSIST_DIR, "last_stock.csv"))
if reviews_up is not None:
    save_file(reviews_up, os.path.join(PERSIST_DIR, "last_reviews.csv"))

def pick_file(up, demo_path, persist_name):
    if up is not None: return up
    if load_demo and os.path.exists(demo_path): return demo_path
    if use_last and os.path.exists(os.path.join(PERSIST_DIR, persist_name)): return os.path.join(PERSIST_DIR, persist_name)
    return None

sales_file = pick_file(sales_up, DEMO_SALES, "last_sales.csv")
stock_file = pick_file(stock_up, DEMO_STOCK, "last_stock.csv")
reviews_file = pick_file(reviews_up, DEMO_REVIEWS, "last_reviews.csv")

if sales_file is None:
    st.info("Upload sales CSV or include demo files in data/ and click Load demo.")
    st.stop()

sales_df, err = safe_read(sales_file)
if sales_df is None:
    st.error(f"Read error: {err}")
    st.stop()
try:
    sales_df = normalize_sales_df(sales_df)
except Exception as e:
    st.error(f"Normalize error: {e}")
    st.stop()

stock_df = None
if stock_file is not None:
    stock_df, _ = safe_read(stock_file)
reviews_df = None
if reviews_file is not None:
    reviews_df, _ = safe_read(reviews_file)

sales_df['Date'] = pd.to_datetime(sales_df['Date'], errors='coerce')
min_date, max_date = sales_df['Date'].min(), sales_df['Date'].max()

# Dashboard with richer charts
if page == "Dashboard":
    st.title("Dashboard — Trends & Insights")
    colf, colp = st.columns([3,1])
    with colf:
        date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    with colp:
        prod_options = ["All"] + sorted(sales_df['Product'].unique().tolist())
        prod_filter = st.selectbox("Product", prod_options)
    sdf = sales_df.copy()
    s,e = date_range
    sdf = sdf[(sdf['Date'] >= pd.to_datetime(s)) & (sdf['Date'] <= pd.to_datetime(e))]
    if prod_filter != "All":
        sdf = sdf[sdf['Product'] == prod_filter]
    total_sales = sdf['Sales'].sum()
    avg_daily = sdf.groupby('Date')['Sales'].sum().mean() if not sdf.empty else 0
    top_product = sdf.groupby('Product')['Sales'].sum().idxmax() if not sdf.empty else "N/A"
    k1,k2,k3 = st.columns(3)
    k1.metric("Total sales", f"${total_sales:,.0f}")
    k2.metric("Avg daily sales", f"{avg_daily:.2f}")
    k3.metric("Top product", top_product)
    st.markdown("---")
    c1,c2 = st.columns([2,1])
    with c1:
        st.subheader("Sales trend + MA7")
        daily = sdf.groupby('Date')['Sales'].sum().reset_index().sort_values('Date')
        if daily.empty:
            st.info("No sales in range.")
        else:
            daily['MA_7'] = daily['Sales'].rolling(window=7, min_periods=1).mean()
            chart = alt.Chart(daily).mark_area(opacity=0.4).encode(x='Date:T', y='Sales:Q')
            line = alt.Chart(daily).mark_line(color='steelblue').encode(x='Date:T', y='Sales:Q')
            ma = alt.Chart(daily).mark_line(color='orange', strokeDash=[4,4]).encode(x='Date:T', y='MA_7:Q')
            st.altair_chart((chart + line + ma).properties(height=360), use_container_width=True)
            # PNG download
            buf = BytesIO()
            fig, ax = plt.subplots(figsize=(10,3))
            ax.plot(daily['Date'], daily['Sales'], label='Sales')
            ax.plot(daily['Date'], daily['MA_7'], label='MA7', linestyle='--')
            ax.set_title('Sales trend + MA7')
            ax.legend()
            fig.tight_layout()
            fig.savefig(buf, format='png')
            buf.seek(0)
            st.download_button("Download trend PNG", data=buf, file_name="sales_trend_ma.png", mime="image/png")
    with c2:
        st.subheader("Cumulative sales & Top products")
        cum = daily.copy() if not daily.empty else pd.DataFrame({'Date':[], 'Sales':[]})
        if not cum.empty:
            cum['Cumulative'] = cum['Sales'].cumsum()
            st.line_chart(cum.set_index('Date')['Cumulative'])
        top10 = sdf.groupby('Product')['Sales'].sum().reset_index().sort_values('Sales', ascending=False).head(10)
        st.table(top10)
    st.markdown("---")
    st.subheader("Monthly sales pivot (Product x YYYY-MM)")
    heat = sales_df.copy()
    heat['YearMonth'] = heat['Date'].dt.to_period('M').astype(str)
    heatf = heat.groupby(['YearMonth','Product'])['Sales'].sum().reset_index()
    if heatf.empty:
        st.write("No data for pivot.")
    else:
        pivot = heatf.pivot(index='Product', columns='YearMonth', values='Sales').fillna(0).astype(int)
        st.dataframe(pivot)

# Inventory page: stock and recommendations
elif page == "Inventory":
    st.title("Inventory — Stock & Recommendations")
    if stock_df is None:
        st.info("Upload stock CSV to view stock table and recommendations.")
    else:
        st.subheader("Stock table")
        st.dataframe(stock_df)
        st.download_button("Download stock CSV", data=stock_df.to_csv(index=False).encode('utf-8'), file_name="stock_export.csv", mime="text/csv")
    st.markdown("---")
    st.subheader("Stock action recommendations")
    prod_list = sorted(sales_df['Product'].unique().tolist())
    sel = st.multiselect("Products to evaluate", options=prod_list, default=prod_list[:8])
    # check for Prophet
    use_prophet = False
    try:
        from prophet import Prophet
        use_prophet = True
    except Exception:
        use_prophet = False
    recs = []
    for p in sel:
        ts = sales_df[sales_df['Product'] == p].groupby('Date')['Sales'].sum().reset_index().sort_values('Date')
        recent_avg = ts['Sales'].tail(30).mean() if len(ts) >= 7 else (ts['Sales'].mean() if len(ts)>0 else 0.0)
        if use_prophet and not ts.empty:
            try:
                m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                dfp = ts.rename(columns={'Date':'ds','Sales':'y'})[['ds','y']]
                m.fit(dfp)
                future = m.make_future_dataframe(periods=30)
                fc = m.predict(future)
                last_date = ts['Date'].max()
                fc_future = fc[fc['ds'] > last_date].head(30)
                forecast = float(fc_future['yhat'].sum())
            except Exception:
                forecast = float(recent_avg * 30)
        else:
            forecast = float(recent_avg * 30)
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
                cur_stock = float(pd.to_numeric(match['Stock'].iloc[0], errors='coerce'))
        adj = 1.0
        if reviews_df is not None:
            r = reviews_df.copy()
            if 'VADER_compound' in r.columns:
                avg_v = r[r['Product']==p]['VADER_compound'].mean() if not r[r['Product']==p].empty else np.nan
                if not np.isnan(avg_v):
                    if avg_v < -0.2: adj = 0.85
                    elif avg_v > 0.3: adj = 1.15
        adjusted = forecast * adj
        safety = 7 * (recent_avg if recent_avg>0 else 1)
        if cur_stock is None or np.isnan(cur_stock):
            action = "No stock data"
        else:
            if cur_stock < adjusted * 0.9 or cur_stock < safety:
                action = "Stock Up"
            elif cur_stock > adjusted * 1.5:
                action = "Reduce"
            else:
                action = "Hold"
        days_left = None
        if recent_avg>0 and cur_stock is not None and not np.isnan(cur_stock):
            days_left = cur_stock / recent_avg
        recs.append({
            "Product": p,
            "RecentAvgDaily": round(recent_avg,2),
            "Forecast30d": round(forecast,2),
            "AdjFactor": round(adj,2),
            "AdjustedForecast": round(adjusted,2),
            "CurrentStock": int(cur_stock) if cur_stock is not None and not np.isnan(cur_stock) else None,
            "DaysLeft": round(days_left,1) if days_left is not None else None,
            "Action": action
        })
    rec_df = pd.DataFrame(recs)
    st.dataframe(rec_df)
    st.download_button("Download stock recommendations CSV", data=rec_df.to_csv(index=False).encode('utf-8'), file_name="stock_recommendations.csv", mime="text/csv")

# Product Performance page
elif page == "Product Performance":
    st.title("Product Performance — Reviews & Suggestions")
    prod_list = ["All"] + sorted(sales_df['Product'].unique().tolist())
    prod = st.selectbox("Choose product", prod_list)
    use_free = st.checkbox("Use free LLM (rule-based)", value=True)
    use_openai = st.checkbox("Use OpenAI (optional)", value=False)
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
        st.subheader("Sample reviews")
        st.write(rprod[['Date','Product','ReviewText']].head(30))
        st.markdown("---")
        st.subheader("Suggestions")
        if use_free:
            texts = [t.lower() for t in rprod['ReviewText'].astype(str).tolist() if t.strip()]
            if not texts:
                st.write("No reviews available.")
            else:
                negatives = [t for t in texts if any(w in t for w in ['broken','defect','overheat','delay','refund','return','compat'])]
                st.write(f"Analyzed {len(texts)} reviews — {len(negatives)} complaint-like.")
                if negatives:
                    st.write("Top complaints:")
                    for ex in negatives[:6]:
                        st.write("- ", ex[:200])
                    if any('overheat' in n for n in negatives):
                        st.write("- Action: investigate thermal issues and update docs.")
                    if any('broken' in n or 'defect' in n for n in negatives):
                        st.write("- Action: audit QC and consider holding restock.")
                else:
                    st.write("No frequent complaints detected. Consider promoting positives.")
        elif use_openai:
            key = st.session_state.get('session_openai_key') or st.secrets.get('OPENAI_API_KEY', None)
            if not key:
                st.error("OpenAI key not available. Set in Settings or st.secrets.")
            else:
                try:
                    import openai
                    openai.api_key = key
                    sample = "\\n".join(rprod['ReviewText'].astype(str).head(40).tolist())
                    prompt = f\"\"\"You are a concise product ops analyst. Given product: {prod} and customer reviews below, list 3 prioritized improvements. Reviews:\\n{sample}\"\"\"
                    resp = openai.ChatCompletion.create(model='gpt-4o-mini', messages=[{'role':'user','content':prompt}], max_tokens=250)
                    suggestion = resp['choices'][0]['message']['content'].strip()
                    st.markdown(suggestion)
                except Exception as e:
                    st.error(f"OpenAI call failed: {e}")

# Reports page
elif page == "Reports":
    st.title("Reports — Forecasts & Sentiment")
    prod = st.selectbox("Choose product for forecast", ["All"] + sorted(sales_df['Product'].unique().tolist()))
    if prod == "All":
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
            future = m.make_future_dataframe(periods=30)
            fc = m.predict(future)
            fig = m.plot(fc)
            st.pyplot(fig)
            st.dataframe(fc[['ds','yhat','yhat_lower','yhat_upper']].tail(30))
        else:
            recent_avg = ts['Sales'].tail(30).mean() if len(ts)>=7 else ts['Sales'].mean()
            future_dates = pd.date_range(start=ts['Date'].max()+pd.Timedelta(days=1), periods=30, freq='D')
            fc = pd.DataFrame([{'ds':d,'yhat':recent_avg} for d in future_dates])
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

# Settings
elif page == "Settings":
    st.title("Settings")
    st.subheader("Session OpenAI key (optional)")
    key = st.text_input("OpenAI API key (session only)", type="password")
    if key:
        st.session_state['session_openai_key'] = key
        st.success("Session key set.")

else:
    st.write("Unknown page")
