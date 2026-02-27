import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load CSVs & Build SQLite Database ─────────────────────────────────────
conn = sqlite3.connect('data/pharma_sales.db')

df_monthly = pd.read_csv('data/raw/salesmonthly.csv', parse_dates=['datum'])
df_daily   = pd.read_csv('data/raw/salesdaily.csv',   parse_dates=['datum'])
df_weekly  = pd.read_csv('data/raw/salesweekly.csv',  parse_dates=['datum'])
df_hourly  = pd.read_csv('data/raw/saleshourly.csv',  parse_dates=['datum'])

drugs  = ['M01AB','M01AE','N02BA','N02BE','N05B','N05C','R03','R06']
labels = {'M01AB':'Diclofenac','M01AE':'Ibuprofen','N02BA':'Aspirin',
          'N02BE':'Paracetamol','N05B':'Anxiolytics','N05C':'Hypnotics',
          'R03':'Salbutamol','R06':'Antihistamines'}

# Melt to long format and save all 4 tables into SQLite
for name, df in [('monthly', df_monthly), ('daily', df_daily),
                 ('weekly', df_weekly),   ('hourly', df_hourly)]:
    df_long = df.melt(id_vars='datum', value_vars=drugs,
                      var_name='drug', value_name='sales')
    df_long['drug_name'] = df_long['drug'].map(labels)
    df_long.to_sql(f'sales_{name}', conn, if_exists='replace', index=False)

print("✅ Database created: data/pharma_sales.db")
print(f"   Tables: sales_monthly, sales_daily, sales_weekly, sales_hourly\n")

# ── SQL Query 1: Total Sales Ranking ──────────────────────────────────────
q1 = """
SELECT 
    drug_name,
    ROUND(SUM(sales), 0)  AS total_sales,
    ROUND(AVG(sales), 2)  AS avg_monthly_sales,
    ROUND(MAX(sales), 2)  AS peak_sales,
    ROUND(MIN(sales), 2)  AS min_sales
FROM sales_monthly
GROUP BY drug_name
ORDER BY total_sales DESC
"""
df_q1 = pd.read_sql(q1, conn)
print("── Q1: Total Sales Ranking ──────────────────────────")
print(df_q1.to_string(index=False))

# ── SQL Query 2: Year-over-Year Growth ────────────────────────────────────
q2 = """
WITH yearly AS (
    SELECT 
        SUBSTR(datum, 1, 4) AS year,
        drug_name,
        SUM(sales) AS annual_sales
    FROM sales_monthly
    GROUP BY year, drug_name
)
SELECT 
    year,
    drug_name,
    ROUND(annual_sales, 0) AS annual_sales,
    ROUND(annual_sales - LAG(annual_sales) OVER 
          (PARTITION BY drug_name ORDER BY year), 0) AS change,
    ROUND((annual_sales - LAG(annual_sales) OVER 
          (PARTITION BY drug_name ORDER BY year))
          / LAG(annual_sales) OVER 
          (PARTITION BY drug_name ORDER BY year) * 100, 1) AS yoy_pct
FROM yearly
ORDER BY drug_name, year
"""
df_q2 = pd.read_sql(q2, conn)
print("\n── Q2: Year-over-Year Growth ────────────────────────")
print(df_q2.to_string(index=False))

# ── SQL Query 3: Quarterly Market Share ───────────────────────────────────
q3 = """
WITH quarterly AS (
    SELECT
        SUBSTR(datum, 1, 4) AS year,
        CASE 
            WHEN CAST(SUBSTR(datum, 6, 2) AS INT) BETWEEN 1 AND 3  THEN 'Q1'
            WHEN CAST(SUBSTR(datum, 6, 2) AS INT) BETWEEN 4 AND 6  THEN 'Q2'
            WHEN CAST(SUBSTR(datum, 6, 2) AS INT) BETWEEN 7 AND 9  THEN 'Q3'
            ELSE 'Q4'
        END AS quarter,
        drug_name,
        SUM(sales) AS qtr_sales
    FROM sales_monthly
    GROUP BY year, quarter, drug_name
)
SELECT
    year, quarter, drug_name,
    ROUND(qtr_sales, 0) AS qtr_sales,
    ROUND(qtr_sales * 100.0 / SUM(qtr_sales) OVER 
          (PARTITION BY year, quarter), 1) AS market_share_pct
FROM quarterly
ORDER BY year, quarter, market_share_pct DESC
"""
df_q3 = pd.read_sql(q3, conn)
print("\n── Q3: Quarterly Market Share ───────────────────────")
print(df_q3.tail(24).to_string(index=False))

# ── SQL Query 4: Best & Worst Month per Drug ──────────────────────────────
q4 = """
SELECT
    drug_name,
    MAX(sales) AS best_month_sales,
    MIN(sales) AS worst_month_sales,
    ROUND(AVG(sales), 1) AS avg_sales,
    ROUND((MAX(sales) - MIN(sales)) / AVG(sales) * 100, 1) AS volatility_pct
FROM sales_monthly
GROUP BY drug_name
ORDER BY volatility_pct DESC
"""
df_q4 = pd.read_sql(q4, conn)
print("\n── Q4: Sales Volatility by Drug ─────────────────────")
print(df_q4.to_string(index=False))

# ── Export results to CSV ─────────────────────────────────────────────────
df_q1.to_csv('data/exports/sql_total_ranking.csv',    index=False)
df_q2.to_csv('data/exports/sql_yoy_growth.csv',       index=False)
df_q3.to_csv('data/exports/sql_market_share.csv',     index=False)
df_q4.to_csv('data/exports/sql_volatility.csv',       index=False)
print("\n✅ All query results saved to data/exports/")

# ── Visualise Q2: YoY Growth Heatmap ─────────────────────────────────────
PALETTE  = ['#1F4E79','#2E75B6','#70AD47','#ED7D31',
            '#FFC000','#9E480E','#5A5A5A','#4BACC6']

df_pivot = df_q2.pivot(index='drug_name', columns='year', values='yoy_pct')
fig, ax = plt.subplots(figsize=(12, 5), facecolor='#F7F9FC')
sns.heatmap(df_pivot, ax=ax, cmap='RdYlGn', center=0,
            annot=True, fmt='.1f', annot_kws={'size': 10},
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'YoY Growth %'})
ax.set_title('Year-over-Year Sales Growth % by Drug (SQL Query)',
             fontsize=14, fontweight='bold', color='#1F4E79', pad=12)
ax.set_xlabel('Year'); ax.set_ylabel('')
ax.tick_params(axis='y', rotation=0)
plt.tight_layout()
plt.savefig('data/exports/sql_yoy_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Visualise Q3: Market Share Over Time ──────────────────────────────────
df_q3['period'] = df_q3['year'].astype(str) + '-' + df_q3['quarter']
df_share_pivot  = df_q3.pivot_table(index='period', columns='drug_name',
                                     values='market_share_pct', aggfunc='mean')
fig, ax = plt.subplots(figsize=(14, 6), facecolor='#F7F9FC')
df_share_pivot.plot(kind='bar', stacked=True, ax=ax,
                    colormap='tab20', edgecolor='white', linewidth=0.4)
ax.set_title('Quarterly Market Share % by Drug (SQL Query)',
             fontsize=14, fontweight='bold', color='#1F4E79', pad=12)
ax.set_xlabel('Quarter'); ax.set_ylabel('Market Share (%)')
ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc='upper left')
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('data/exports/sql_market_share.png', dpi=150, bbox_inches='tight')
plt.show()

conn.close()
print("\n✅ Step 2 complete!")