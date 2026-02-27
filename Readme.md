# Pharmaceutical Sales Performance Analysis

A complete end-to-end data analytics project analyzing pharmaceutical 
sales data using Python, SQL, and Power BI.

## Project Structure
pharma-sales-analysis/
├── data/
│   ├── raw/          # Source CSV files
│   └── exports/      # Generated outputs
├── src/
│   ├── 01_eda.py           # Exploratory Data Analysis
│   ├── 02_sql.py           # SQL Analysis & Database
│   ├── 03_seasonality.py   # Seasonal Decomposition
│   └── 04_forecast.py      # SARIMA Forecasting
├── requirements.txt
└── README.md

## Tools & Technologies
- Python (pandas, matplotlib, seaborn, statsmodels)
- SQL (SQLite)
- Power BI Desktop

## Dataset
- Source: Kaggle — Pharma Sales Data by Milan Zdravkovic
- Period: January 2014 – October 2019
- 8 drug categories, 4 granularities (hourly/daily/weekly/monthly)

## Key Findings
- Paracetamol dominates with 49% market share
- Aspirin identified as AT-RISK (-10.6% forecast decline)
- Antihistamines fastest growing at +25.9%
- 2018 was peak year (+36% avg growth)
- Respiratory drugs peak in December (winter season)

## Analysis Steps
1. **EDA** — Sales trends, market share, YoY comparison
2. **SQL Analysis** — Database queries for KPIs and growth metrics
3. **Seasonality** — Decomposition, seasonal index, hourly patterns
4. **Forecasting** — SARIMA models to predict next 12 months
5. **Power BI** — 3-page interactive dashboard

## How to Run
# Clone the repo
git clone https://github.com/YOURUSERNAME/pharma-sales-analysis.git
cd pharma-sales-analysis

# Install dependencies
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run scripts in order
python src/01_eda.py
python src/02_sql.py
python src/03_seasonality.py
python src/04_forecast.py
