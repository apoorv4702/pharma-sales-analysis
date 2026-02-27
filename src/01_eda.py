import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Load Data ─────────────────────────────────────────────────────────────
df_m = pd.read_csv('data/raw/salesmonthly.csv', parse_dates=['datum'])
df_d = pd.read_csv('data/raw/salesdaily.csv',   parse_dates=['datum'])

drugs  = ['M01AB','M01AE','N02BA','N02BE','N05B','N05C','R03','R06']
labels = {'M01AB':'Diclofenac','M01AE':'Ibuprofen','N02BA':'Aspirin',
          'N02BE':'Paracetamol','N05B':'Anxiolytics','N05C':'Hypnotics',
          'R03':'Salbutamol','R06':'Antihistamines'}

# Long format
df_long = df_m.melt(id_vars='datum', value_vars=drugs,
                    var_name='drug', value_name='sales')
df_long['drug_label'] = df_long['drug'].map(labels)
df_long['year']  = df_long['datum'].dt.year
df_long['month'] = df_long['datum'].dt.month

PALETTE  = ['#1F4E79','#2E75B6','#70AD47','#ED7D31',
            '#FFC000','#9E480E','#5A5A5A','#4BACC6']
drug_clr = dict(zip(drugs, PALETTE))

# ── Figure 1: Overview Dashboard ──────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor='#F7F9FC')
fig.suptitle('Pharmaceutical Sales — EDA Dashboard',
             fontsize=17, fontweight='bold', color='#1F4E79', y=0.98)

# Chart 1: Monthly trends
ax = axes[0, 0]
for drug in drugs:
    sub = df_long[df_long['drug'] == drug].sort_values('datum')
    ax.plot(sub['datum'], sub['sales'], label=labels[drug],
            color=drug_clr[drug], linewidth=1.8)
ax.set_title('Monthly Sales Trends (2014–2019)', fontweight='bold', color='#1F4E79')
ax.set_xlabel('Date'); ax.set_ylabel('Sales Units')
ax.legend(fontsize=7.5, ncol=2); ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines[['top','right']].set_visible(False)

# Chart 2: Total sales bar
ax = axes[0, 1]
totals = df_long.groupby('drug')['sales'].sum().reindex(drugs)
bars = ax.bar([labels[d] for d in drugs], totals.values,
              color=PALETTE, edgecolor='white')
ax.set_title('Total Sales by Drug (All Years)', fontweight='bold', color='#1F4E79')
ax.set_ylabel('Total Sales Units')
ax.tick_params(axis='x', rotation=35)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines[['top','right']].set_visible(False)
for bar, val in zip(bars, totals.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{val:,.0f}', ha='center', fontsize=8, fontweight='bold')

# Chart 3: Market share pie
ax = axes[1, 0]
ax.pie(totals.values, labels=[labels[d] for d in drugs], colors=PALETTE,
       autopct='%1.1f%%', startangle=140, pctdistance=0.78,
       wedgeprops=dict(edgecolor='white', linewidth=1.5))
ax.set_title('Market Share by Drug Category', fontweight='bold', color='#1F4E79')

# Chart 4: YoY grouped bar
ax = axes[1, 1]
yearly = df_long.groupby(['year','drug'])['sales'].sum().unstack()[drugs]
x, w = np.arange(len(yearly)), 0.1
for i, drug in enumerate(drugs):
    ax.bar(x + i*w - 0.35, yearly[drug].values, width=w,
           label=labels[drug], color=drug_clr[drug], edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(yearly.index)
ax.set_title('Annual Sales by Drug (YoY)', fontweight='bold', color='#1F4E79')
ax.legend(fontsize=7, ncol=2); ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('data/exports/eda_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Figure 2: Deep Dive ───────────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(18, 12), facecolor='#F7F9FC')
fig2.suptitle('Deep Dive — Growth, Seasonality & Market Dynamics',
              fontsize=17, fontweight='bold', color='#1F4E79', y=0.98)

# Chart 5: YoY growth heatmap
ax = axes2[0, 0]
yearly_pct = yearly.pct_change() * 100
sns.heatmap(yearly_pct.T.rename(index=labels), ax=ax, cmap='RdYlGn',
            center=0, annot=True, fmt='.1f', annot_kws={'size':9},
            linewidths=0.5, linecolor='white')
ax.set_title('YoY Growth % by Drug', fontweight='bold', color='#1F4E79')
ax.tick_params(axis='y', rotation=0)

# Chart 6: Stacked area market share
ax = axes2[0, 1]
share = df_m.set_index('datum')[drugs]
share_pct = share.div(share.sum(axis=1), axis=0) * 100
ax.stackplot(share_pct.index, [share_pct[d] for d in drugs],
             labels=[labels[d] for d in drugs], colors=PALETTE, alpha=0.85)
ax.set_title('Market Share Over Time (%)', fontweight='bold', color='#1F4E79')
ax.set_ylabel('Market Share (%)'); ax.legend(fontsize=7.5, ncol=2)
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.spines[['top','right']].set_visible(False)

# Chart 7: Seasonality heatmap
ax = axes2[1, 0]
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
seasonal = df_long.groupby(['month','drug'])['sales'].mean().unstack()[drugs]
seasonal.index = month_names
sns.heatmap(seasonal.T.rename(index=labels), ax=ax, cmap='Blues',
            annot=True, fmt='.0f', annot_kws={'size':8},
            linewidths=0.4, linecolor='white')
ax.set_title('Avg Monthly Sales — Seasonality Heatmap', fontweight='bold', color='#1F4E79')
ax.tick_params(axis='y', rotation=0)

# Chart 8: Rolling 3M average
ax = axes2[1, 1]
for drug in ['N02BE','N05B','R03','M01AB']:
    sub = df_m.set_index('datum')[drug].rolling(3).mean()
    ax.plot(sub.index, sub.values, label=labels[drug],
            color=drug_clr[drug], linewidth=2)
ax.set_title('Rolling 3-Month Average — Key Drugs', fontweight='bold', color='#1F4E79')
ax.legend(fontsize=9); ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('data/exports/eda_deepdive.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Print Summary Stats ───────────────────────────────────────────────────
print("\n===== KEY FINDINGS =====")
print(f"\nTop Drug:    {labels[totals.idxmax()]} ({totals.max():,.0f} units)")
print(f"Lowest Drug: {labels[totals.idxmin()]} ({totals.min():,.0f} units)")
print("\nAvg YoY Growth by Year:")
print(yearly_pct.mean(axis=1).round(2).to_string())
print("\nPeak Month per Drug:")
for d in drugs:
    print(f"  {labels[d]:18s}: {seasonal[d].idxmax()}")