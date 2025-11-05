# 🧭 Inventory Management System

An interactive **Streamlit-based Inventory Management System** designed to help businesses analyze sales, manage stock, forecast demand, and assess customer feedback.  
This dashboard integrates **data-driven insights**, **sentiment analysis**, and **LLM-based review feedback generation** — all in one app.

---

## 🚀 Features

### 🔹 Dashboard
- View **total sales**, **average daily sales**, and **country-level trends**.
- Interactive **date range and product filters**.
- Visualize **sales trends** and identify **best-selling products**.

### 🔹 Inventory
- Manage and monitor stock levels.
- Detect **low-stock items** based on recent sales.
- Export and update inventory data.

### 🔹 Product Performance
- Analyze **customer reviews** and automatically generate **improvement suggestions**.
- Choose between:
  - 🧠 **Free LLM** (built-in rule-based analyzer) – no API key required.
  - 🤖 **OpenAI LLM** (optional) – uses GPT-4o-mini if `OPENAI_API_KEY` is available.
- Detect common complaint patterns and improvement areas automatically.

### 🔹 Reports
- Perform **sales forecasting** (using Prophet or fallback model).
- Conduct **sentiment analysis** on reviews with NLTK VADER.
- Export forecasts, enriched reviews, and recommendations.

### 🔹 Settings
- Configure **OpenAI API keys** (either via Streamlit Secrets or session input).
- Restore original uploads or reload demo data.

---

## 📂 Project Structure

```
Inventory-Management-System/
│
├── app.py                             # Main Streamlit application
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
│
├── data/                              # Demo data folder (included in repo)
│   ├── demo_sales_dataset.csv
│   ├── demo_stock_dataset.csv
│   ├── demo_reviews_dataset.csv
│   └── aic_logo.png                   # Sidebar logo
│
└── .streamlit/
    └── secrets.toml                   # (optional) for storing API keys
```

---

## 🛠️ Installation & Local Run

### 1. Clone this repository
```bash
git clone https://github.com/<your-username>/Inventory-Management-System.git
cd Inventory-Management-System
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

### Steps
1. Push this repository to your GitHub account.
2. Go to [https://share.streamlit.io](https://share.streamlit.io).
3. Select **New App → Connect GitHub → Select Repo**.
4. Choose:
   - **Branch:** `main` (or your default branch)
   - **File path:** `app.py`
5. Click **Deploy** 🚀

### Secrets Configuration (Optional)
In your Streamlit Cloud app:
- Navigate to **Settings → Secrets → Add secrets**
- Add the following line:
  ```toml
  OPENAI_API_KEY = "sk-..."
  ```

---

## 📊 Data Requirements

### **1️⃣ Sales CSV**
Required columns:
- `Date` – transaction date
- `Product` – product name
- `Sales` – numeric sales amount
- *(optional)* `Country` – country/region

### **2️⃣ Stock CSV**
Required columns:
- `Product` – product name
- `Stock` – current stock units

### **3️⃣ Reviews CSV**
Required columns:
- `Product` – product name
- `Date` – review date (optional)
- `ReviewText` – text of review
- *(optional)* `Rating` – numeric rating (1–5)

---

## 🧠 LLM Options

| Mode | Description | API Needed |
|------|--------------|-------------|
| **Free LLM** | Built-in rule-based analyzer that extracts complaint snippets and suggests improvements. | ❌ No |
| **OpenAI LLM** | GPT-4o-mini powered suggestion engine for product improvement insights. | ✅ Yes (`OPENAI_API_KEY`) |

Example insights:
> "Investigate cooling system — multiple reviews mention overheating issues."  
> "Customers appreciate performance; promote stability features."

---

## 🧩 Technologies Used
- **Streamlit** – dashboard & UI
- **Pandas / NumPy** – data handling
- **Altair / Matplotlib** – visualization
- **NLTK (VADER)** – sentiment analysis
- **Prophet** – forecasting
- **OpenAI API** – optional LLM-based suggestions

---

## 🧾 License
This project is open-source and free to use for educational and personal purposes.

---

## 👤 Author
Developed by **Bhoomika K M**  
B.Tech AIML | CMR University  
📧 *You can add your email or GitHub profile link here.*

---

## 🌟 Acknowledgments
Special thanks to:
- Streamlit team for the easy deployment.
- OpenAI for LLM integration capabilities.
- NLTK & Prophet developers for amazing open-source tools.
