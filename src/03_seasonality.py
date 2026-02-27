import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

# ── Load Data ─────────────────────────────────────────────────────────────
df_m = pd.read_csv('data/raw/salesmonthly.csv', parse_dates=['datum'])
df_m = df_m.set_index('datum')
df_d = pd.read_csv('data/raw/salesdaily.csv',   parse_dates=['datum'])
df_h = pd.read_csv('data/raw/saleshourly.csv',  parse_dates=['datum'])

drugs  = ['M01AB','M01AE','N02BA','N02BE','N05B','N05C','R03','R06']
labels = {'M01AB':'Diclofenac','M01AE':'Ibuprofen','N02BA':'Aspirin',
          'N02BE':'Paracetamol','N05B':'Anxiolytics','N05C':'Hypnotics',
          'R03':'Salbutamol','R06':'Antihistamines'}

PALETTE  = ['#1F4E79','#2E75B6','#70AD47','#ED7D31',
            '#FFC000','#9E480E','#5A5A5A','#4BACC6']
drug_clr = dict(zip(drugs, PALETTE))
month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

# ── Key fix: resample to month-start to avoid nulls ───────────────────────
def get_series(drug):
    return df_m[drug].resample('MS').sum()

# ════════════════════════════════════════════════════════════════════════════
# CHART 1 — Seasonal Decomposition for Top 4 Drugs
# ════════════════════════════════════════════════════════════════════════════
top4 = ['N02BE','N05B','R03','M01AB']

fig, axes = plt.subplots(4, 4, figsize=(22, 16), facecolor='#F7F9FC')
fig.suptitle('Seasonal Decomposition — Top 4 Drugs\n(Observed | Trend | Seasonal | Residual)',
             fontsize=15, fontweight='bold', color='#1F4E79', y=0.99)

for row, drug in enumerate(top4):
    series = get_series(drug)
    result = seasonal_decompose(series, model='additive', period=12)
    components = [result.observed, result.trend, result.seasonal, result.resid]
    comp_names = ['Observed', 'Trend', 'Seasonal', 'Residual']
    colors     = [drug_clr[drug], '#2E75B6', '#ED7D31', '#888888']

    for col, (comp, cname, color) in enumerate(zip(components, comp_names, colors)):
        ax = axes[row, col]
        ax.set_facecolor('#FAFBFD')
        ax.plot(comp.index, comp.values, color=color, linewidth=1.6)
        if col == 0:
            ax.set_ylabel(labels[drug], fontweight='bold',
                          color=drug_clr[drug], fontsize=10)
        if row == 0:
            ax.set_title(cname, fontweight='bold', color='#1F4E79', fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('data/exports/seasonality_decomposition.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart 1 saved — Seasonal Decomposition")

# ════════════════════════════════════════════════════════════════════════════
# CHART 2 — Seasonal Index per Drug
# ════════════════════════════════════════════════════════════════════════════
seasonal_indices = {}
for drug in drugs:
    series = get_series(drug)
    result = seasonal_decompose(series, model='additive', period=12)
    idx = result.seasonal.groupby(result.seasonal.index.month).mean()
    seasonal_indices[labels[drug]] = idx.values

df_si = pd.DataFrame(seasonal_indices, index=month_names).T

fig, axes = plt.subplots(1, 2, figsize=(20, 7), facecolor='#F7F9FC')
fig.suptitle('Seasonal Index Analysis — Which Months Drive Sales?',
             fontsize=15, fontweight='bold', color='#1F4E79')

# Heatmap
ax = axes[0]
sns.heatmap(df_si, ax=ax, cmap='RdYlGn', center=1.0,
            annot=True, fmt='.2f', annot_kws={'size': 9},
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Seasonal Index (1.0 = average)'})
ax.set_title('Seasonal Index Heatmap\n(>1.0 = above average, <1.0 = below average)',
             fontweight='bold', color='#1F4E79', pad=10)
ax.tick_params(axis='y', rotation=0)

# Bar chart
ax = axes[1]
ax.set_facecolor('#FAFBFD')
x = np.arange(12)
w = 0.2
key_drugs = ['Paracetamol','Salbutamol','Anxiolytics','Antihistamines']
key_clrs  = ['#1F4E79','#70AD47','#ED7D31','#4BACC6']
for i, (drug, color) in enumerate(zip(key_drugs, key_clrs)):
    ax.bar(x + i*w - 0.3, df_si.loc[drug], width=w,
           label=drug, color=color, edgecolor='white', alpha=0.9)
ax.axhline(1.0, color='red', linestyle='--', linewidth=1.2, label='Baseline (1.0)')
ax.set_xticks(x); ax.set_xticklabels(month_names)
ax.set_title('Seasonal Index — Key Drugs by Month',
             fontweight='bold', color='#1F4E79', pad=10)
ax.set_ylabel('Seasonal Index')
ax.legend(fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('data/exports/seasonal_index.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart 2 saved — Seasonal Index")

# ════════════════════════════════════════════════════════════════════════════
# CHART 3 — Day of Week & Hour of Day Patterns
# ════════════════════════════════════════════════════════════════════════════
df_d['Weekday Name'] = pd.Categorical(
    df_d['Weekday Name'],
    categories=['Monday','Tuesday','Wednesday','Thursday',
                'Friday','Saturday','Sunday'],
    ordered=True)
dow    = df_d.groupby('Weekday Name')[drugs].mean()
hourly = df_h.groupby('Hour')[drugs].mean()

fig, axes = plt.subplots(1, 2, figsize=(20, 7), facecolor='#F7F9FC')
fig.suptitle('Temporal Patterns — Day of Week & Hour of Day',
             fontsize=15, fontweight='bold', color='#1F4E79')

ax = axes[0]
ax.set_facecolor('#FAFBFD')
for drug, color in zip(drugs, PALETTE):
    ax.plot(dow.index, dow[drug], label=labels[drug],
            color=color, linewidth=2, marker='o', markersize=5)
ax.set_title('Avg Sales by Day of Week', fontweight='bold', color='#1F4E79', pad=10)
ax.set_ylabel('Avg Sales Units')
ax.tick_params(axis='x', rotation=30)
ax.legend(fontsize=8, ncol=2)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines[['top','right']].set_visible(False)

ax = axes[1]
ax.set_facecolor('#FAFBFD')
for drug, color in zip(drugs, PALETTE):
    ax.plot(hourly.index, hourly[drug], label=labels[drug],
            color=color, linewidth=2)
ax.set_title('Avg Sales by Hour of Day', fontweight='bold', color='#1F4E79', pad=10)
ax.set_xlabel('Hour'); ax.set_ylabel('Avg Sales Units')
ax.legend(fontsize=8, ncol=2)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines[['top','right']].set_visible(False)
ax.set_xticks(range(8, 22, 1))

plt.tight_layout()
plt.savefig('data/exports/temporal_patterns.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart 3 saved — Temporal Patterns")

# ── Print Key Insights ────────────────────────────────────────────────────
print("\n===== SEASONAL INSIGHTS =====")
print("\nPeak month per drug:")
for drug in df_si.index:
    peak = df_si.loc[drug].idxmax()
    val  = df_si.loc[drug].max()
    print(f"  {drug:18s}: {peak:>4s}  (index = {val:.2f})")

print("\nLowest month per drug:")
for drug in df_si.index:
    trough = df_si.loc[drug].idxmin()
    val    = df_si.loc[drug].min()
    print(f"  {drug:18s}: {trough:>4s}  (index = {val:.2f})")

df_si.to_csv('data/exports/seasonal_index_table.csv')
print("\n✅ Step 3 Complete!")