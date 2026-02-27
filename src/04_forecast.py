import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error

# ── Load Data ─────────────────────────────────────────────────────────────
df_m = pd.read_csv('data/raw/salesmonthly.csv', parse_dates=['datum'])
df_m = df_m.set_index('datum')

drugs  = ['M01AB','M01AE','N02BA','N02BE','N05B','N05C','R03','R06']
labels = {'M01AB':'Diclofenac','M01AE':'Ibuprofen','N02BA':'Aspirin',
          'N02BE':'Paracetamol','N05B':'Anxiolytics','N05C':'Hypnotics',
          'R03':'Salbutamol','R06':'Antihistamines'}

PALETTE  = ['#1F4E79','#2E75B6','#70AD47','#ED7D31',
            '#FFC000','#9E480E','#5A5A5A','#4BACC6']
drug_clr = dict(zip(drugs, PALETTE))

def get_series(drug):
    return df_m[drug].resample('MS').sum()

# ════════════════════════════════════════════════════════════════════════════
# SARIMA FORECAST — All 8 Drugs
# ════════════════════════════════════════════════════════════════════════════
print("Running SARIMA forecasts — please wait...\n")

results_summary = []
forecast_store  = {}

for drug in drugs:
    series = get_series(drug)
    train  = series[:-6]   # hold out last 6 months for validation
    test   = series[-6:]

    # Fit SARIMA(1,1,1)(1,1,0,12)
    model  = SARIMAX(train,
                     order=(1, 1, 1),
                     seasonal_order=(1, 1, 0, 12),
                     enforce_stationarity=False,
                     enforce_invertibility=False).fit(disp=False)

    # Forecast 6 (test) + 12 (future) = 18 steps
    pred       = model.forecast(steps=18)
    pred_ci    = model.get_forecast(steps=18).conf_int()
    test_pred  = pred[:6]
    future     = pred[6:]

    # Accuracy on test set
    mape = mean_absolute_percentage_error(test, test_pred) * 100

    # Business signal
    current_avg  = train[-12:].mean()
    forecast_avg = future.mean()
    change_pct   = (forecast_avg - current_avg) / current_avg * 100

    if change_pct <= -5:
        status = '⚠️  AT-RISK'
        color  = '#C00000'
    elif change_pct >= 5:
        status = '⬆️  GROWING'
        color  = '#1E6B3C'
    else:
        status = '✅  STABLE'
        color  = '#2E75B6'

    results_summary.append({
        'Drug'          : labels[drug],
        'Current Avg'   : round(current_avg, 1),
        'Forecast Avg'  : round(forecast_avg, 1),
        'Change %'      : round(change_pct, 1),
        'MAPE %'        : round(mape, 1),
        'Status'        : status
    })

    forecast_store[drug] = {
        'series'   : series,
        'train'    : train,
        'test'     : test,
        'future'   : future,
        'pred_ci'  : pred_ci,
        'color'    : color,
        'status'   : status,
        'change'   : round(change_pct, 1)
    }

    print(f"  {labels[drug]:18s} | Change: {change_pct:+.1f}%  | MAPE: {mape:.1f}%  | {status}")

# ════════════════════════════════════════════════════════════════════════════
# CHART 1 — Forecast Plots for All 8 Drugs
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 2, figsize=(20, 20), facecolor='#F7F9FC')
fig.suptitle('SARIMA Sales Forecast — Next 12 Months\nAll Drug Categories',
             fontsize=16, fontweight='bold', color='#1F4E79', y=0.99)

for ax, drug in zip(axes.flat, drugs):
    d      = forecast_store[drug]
    color  = drug_clr[drug]
    ax.set_facecolor('#FAFBFD')

    # Historical
    ax.plot(d['series'].index, d['series'].values,
            color=color, linewidth=2, label='Actual Sales')

    # Test period overlay
    ax.plot(d['test'].index, d['test'].values,
            color='orange', linewidth=2, linestyle='--', label='Validation (actual)')

    # Future forecast
    future_idx = pd.date_range(d['series'].index[-1] + pd.DateOffset(months=1),
                               periods=12, freq='MS')
    ax.plot(future_idx, d['future'].values,
            color=d['color'], linewidth=2.5, linestyle='--', label='Forecast')

    # Confidence interval
    ci = d['pred_ci'].iloc[6:]
    ax.fill_between(future_idx,
                    ci.iloc[:, 0].values,
                    ci.iloc[:, 1].values,
                    color=d['color'], alpha=0.15, label='95% CI')

    # Status badge
    ax.set_title(f"{labels[drug]}  |  {d['status']}  ({d['change']:+.1f}%)",
                 fontweight='bold', color=d['color'], fontsize=10)
    ax.axvline(d['series'].index[-1], color='gray',
               linestyle=':', linewidth=1.2, alpha=0.7)
    ax.legend(fontsize=7)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('data/exports/forecast_all_drugs.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Chart 1 saved — All Drug Forecasts")

# ════════════════════════════════════════════════════════════════════════════
# CHART 2 — Executive Summary: Status Dashboard
# ════════════════════════════════════════════════════════════════════════════
df_summary = pd.DataFrame(results_summary).sort_values('Change %')

fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor='#F7F9FC')
fig.suptitle('Executive Forecast Summary — Resource Allocation Recommendations',
             fontsize=14, fontweight='bold', color='#1F4E79')

# Bar chart — forecast change %
ax = axes[0]
ax.set_facecolor('#FAFBFD')
bar_colors = ['#C00000' if x <= -5 else '#1E6B3C' if x >= 5 else '#2E75B6'
              for x in df_summary['Change %']]
bars = ax.barh(df_summary['Drug'], df_summary['Change %'],
               color=bar_colors, edgecolor='white', height=0.6)
ax.axvline(0, color='black', linewidth=1)
ax.axvline(-5, color='#C00000', linewidth=1, linestyle='--', alpha=0.5)
ax.axvline(5,  color='#1E6B3C', linewidth=1, linestyle='--', alpha=0.5)
ax.set_title('Forecast Change % vs Current Baseline',
             fontweight='bold', color='#1F4E79', pad=10)
ax.set_xlabel('Change %')
for bar, val in zip(bars, df_summary['Change %']):
    ax.text(val + (0.3 if val >= 0 else -0.3), bar.get_y() + bar.get_height()/2,
            f'{val:+.1f}%', va='center',
            ha='left' if val >= 0 else 'right', fontsize=9, fontweight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.4)
ax.spines[['top','right']].set_visible(False)

# Recommendation table
ax = axes[1]
ax.axis('off')
col_labels = ['Drug', 'Curr Avg', 'Fcst Avg', 'Change %', 'MAPE %', 'Status']
table_data = [[r['Drug'], f"{r['Current Avg']:,.0f}", f"{r['Forecast Avg']:,.0f}",
               f"{r['Change %']:+.1f}%", f"{r['MAPE %']:.1f}%", r['Status']]
              for _, r in df_summary.iterrows()]
table = ax.table(cellText=table_data, colLabels=col_labels,
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 2.0)

# Color rows by status
for i, (_, row) in enumerate(df_summary.iterrows()):
    clr = '#FFE0E0' if '⚠️' in row['Status'] else \
          '#E0F4E0' if '⬆️' in row['Status'] else '#E0EEFF'
    for j in range(len(col_labels)):
        table[i+1, j].set_facecolor(clr)
# Header style
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#1F4E79')
    table[0, j].set_text_props(color='white', fontweight='bold')

ax.set_title('Drug Portfolio Status & Recommendations',
             fontweight='bold', color='#1F4E79', pad=20)

plt.tight_layout()
plt.savefig('data/exports/forecast_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart 2 saved — Executive Summary")

# ── Save forecast results ─────────────────────────────────────────────────
df_summary.to_csv('data/exports/forecast_results.csv', index=False)

print("\n===== FORECAST RESULTS =====")
print(df_summary.to_string(index=False))

print("\n===== BUSINESS RECOMMENDATIONS =====")
at_risk = df_summary[df_summary['Status'].str.contains('AT-RISK')]
growing = df_summary[df_summary['Status'].str.contains('GROWING')]
stable  = df_summary[df_summary['Status'].str.contains('STABLE')]

if len(at_risk):
    print(f"\n⚠️  AT-RISK drugs ({len(at_risk)}) — reduce rep coverage, investigate cause:")
    for _, r in at_risk.iterrows():
        print(f"   • {r['Drug']} ({r['Change %']:+.1f}%)")

if len(growing):
    print(f"\n⬆️  GROWING drugs ({len(growing)}) — increase investment & rep allocation:")
    for _, r in growing.iterrows():
        print(f"   • {r['Drug']} ({r['Change %']:+.1f}%)")

if len(stable):
    print(f"\n✅  STABLE drugs ({len(stable)}) — maintain current strategy:")
    for _, r in stable.iterrows():
        print(f"   • {r['Drug']} ({r['Change %']:+.1f}%)")

print("\n✅ Step 4 Complete!")