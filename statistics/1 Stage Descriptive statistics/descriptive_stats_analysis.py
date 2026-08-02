# ============================================================
#  STAGE 1 — DESCRIPTIVE STATISTICS
#  E-commerce Customer Spend Analysis
#  Full Project: Dataset + EDA + Statistics + Visualization
# ============================================================

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor':   '#FFFFFF',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'font.family': 'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

PURPLE  = '#534AB7'
TEAL    = '#0F6E56'
AMBER   = '#BA7517'
CORAL   = '#993C1D'
BLUE    = '#185FA5'
LGRAY   = '#F1EFE8'
DGRAY   = '#5F5E5A'

# ── Load Data ──────────────────────────────────────────────
df = pd.read_csv('ecommerce_orders.csv')
ov = df['order_value_inr']

print("=" * 60)
print("  E-COMMERCE DESCRIPTIVE STATISTICS — FULL ANALYSIS")
print("=" * 60)

# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 1: UNDERSTAND YOUR DATA FIRST                     ║
# ╚══════════════════════════════════════════════════════════╝
print("\n── STEP 1: Dataset Overview ──")
print(f"  Total orders      : {len(df):,}")
print(f"  Columns           : {list(df.columns)}")
print(f"  Missing values    : {df.isnull().sum().sum()}")
print(df.dtypes.to_string())

# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 2: MEASURES OF CENTRAL TENDENCY                   ║
# ╚══════════════════════════════════════════════════════════╝
print("\n── STEP 2: Measures of Central Tendency ──")
mean_val   = ov.mean()
median_val = ov.median()
mode_val   = ov.mode()[0]

print(f"  Mean   (average)  : ₹{mean_val:,.2f}")
print(f"  Median (middle)   : ₹{median_val:,.2f}")
print(f"  Mode   (most freq): ₹{mode_val:,.2f}")
print()
print("  ► WHY MEAN > MEDIAN here?")
print("    Because a few big-spender outliers PULL the mean up.")
print("    Median is the TRUE 'typical order' for pricing strategy.")
diff_pct = ((mean_val - median_val) / median_val) * 100
print(f"    Mean is {diff_pct:.1f}% higher than Median — significant skew!")

# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 3: MEASURES OF SPREAD / DISPERSION                ║
# ╚══════════════════════════════════════════════════════════╝
print("\n── STEP 3: Measures of Spread ──")
std_val      = ov.std()
var_val      = ov.var()
range_val    = ov.max() - ov.min()
cv           = (std_val / mean_val) * 100
q1, q2, q3  = ov.quantile([0.25, 0.50, 0.75])
iqr          = q3 - q1

print(f"  Standard Deviation: ₹{std_val:,.2f}")
print(f"  Variance          : ₹{var_val:,.2f}")
print(f"  Range             : ₹{range_val:,.2f}")
print(f"  Coeff. of Variation: {cv:.1f}%  ← high = wide spread")
print(f"  Q1 (25th pct)     : ₹{q1:,.2f}")
print(f"  Q2 (50th pct)     : ₹{q2:,.2f}")
print(f"  Q3 (75th pct)     : ₹{q3:,.2f}")
print(f"  IQR (Q3 - Q1)     : ₹{iqr:,.2f}")

# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 4: SHAPE — SKEWNESS & KURTOSIS                    ║
# ╚══════════════════════════════════════════════════════════╝
print("\n── STEP 4: Shape of Distribution ──")
skew_val = ov.skew()
kurt_val = ov.kurtosis()

print(f"  Skewness : {skew_val:.4f}")
if skew_val > 1:
    print("    ► Highly RIGHT-SKEWED — long tail of big spenders")
    print("      Business insight: Small % of customers drive big revenue")
elif skew_val > 0.5:
    print("    ► Moderately right-skewed")
else:
    print("    ► Roughly symmetric")

print(f"  Kurtosis : {kurt_val:.4f}")
if kurt_val > 3:
    print("    ► Leptokurtic (heavy tails) — extreme values common")
else:
    print("    ► Near normal tail behavior")

# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 5: OUTLIER DETECTION — IQR METHOD                 ║
# ╚══════════════════════════════════════════════════════════╝
print("\n── STEP 5: Outlier Detection (IQR Method) ──")
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
outliers    = df[(ov < lower_fence) | (ov > upper_fence)]
normal_df   = df[(ov >= lower_fence) & (ov <= upper_fence)]

print(f"  Lower fence: ₹{lower_fence:,.2f}")
print(f"  Upper fence: ₹{upper_fence:,.2f}")
print(f"  Total outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}% of orders)")
print(f"  Outlier revenue: ₹{outliers['order_value_inr'].sum():,.2f}")
print(f"  Outlier % of total revenue: "
      f"{outliers['order_value_inr'].sum()/df['order_value_inr'].sum()*100:.1f}%")
print()
print("  ► Business Insight:")
print(f"    {len(outliers)} customers ({len(outliers)/len(df)*100:.1f}%) "
      f"contribute "
      f"{outliers['order_value_inr'].sum()/df['order_value_inr'].sum()*100:.1f}% of revenue")
print("    → These are your VIP customers. Target them with loyalty programs!")

# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 6: PERCENTILE ANALYSIS                            ║
# ╚══════════════════════════════════════════════════════════╝
print("\n── STEP 6: Percentile Analysis ──")
percentiles = [10, 25, 50, 75, 90, 95, 99]
pct_values  = np.percentile(ov, percentiles)
for p, v in zip(percentiles, pct_values):
    print(f"  P{p:>2}: ₹{v:>10,.2f}")
print()
print(f"  ► Top 10% of orders start at: ₹{np.percentile(ov,90):,.2f}")
print(f"  ► Top 1% (VIP) start at      : ₹{np.percentile(ov,99):,.2f}")
print("    Use P90 as the discount threshold for premium customers.")

# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 7: CATEGORY-WISE BREAKDOWN                        ║
# ╚══════════════════════════════════════════════════════════╝
print("\n── STEP 7: Order Value by Category ──")
cat_stats = df.groupby('category')['order_value_inr'].agg(
    Count='count',
    Mean=lambda x: x.mean(),
    Median='median',
    Std='std',
    P90=lambda x: x.quantile(0.9)
).round(2).sort_values('Mean', ascending=False)
print(cat_stats.to_string())

# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 8: NORMALITY TEST                                 ║
# ╚══════════════════════════════════════════════════════════╝
print("\n── STEP 8: Normality Test (Shapiro-Wilk) ──")
sample_for_test = ov.sample(200, random_state=42)
stat, p = stats.shapiro(sample_for_test)
print(f"  W-statistic: {stat:.4f}")
print(f"  p-value    : {p:.6f}")
print(f"  Result     : {'NOT normal (p<0.05) → use median-based stats' if p < 0.05 else 'Normal distribution'}")

# ╔══════════════════════════════════════════════════════════╗
# ║  FINAL BUSINESS SUMMARY                                 ║
# ╚══════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("  BUSINESS RECOMMENDATIONS")
print("=" * 60)
print(f"  1. Set free-shipping threshold at ₹{median_val*1.5:,.0f} (1.5× median)")
print(f"  2. VIP program: customers spending > ₹{np.percentile(ov,90):,.0f}/order (P90)")
print(f"  3. {len(outliers)} high-value customers need dedicated account managers")
cat_top = cat_stats.index[0]
print(f"  4. '{cat_top}' has highest avg order — prioritize in ad spend")
print(f"  5. Skewness {skew_val:.1f} → don't use mean for KPI dashboards, use median")
print("=" * 60)


# ╔══════════════════════════════════════════════════════════╗
# ║  VISUALIZATION — 6-PANEL DASHBOARD                      ║
# ╚══════════════════════════════════════════════════════════╝
fig = plt.figure(figsize=(18, 14))
fig.suptitle('E-Commerce Descriptive Statistics Dashboard',
             fontsize=18, fontweight='bold', color=DGRAY, y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

# Panel 1: Histogram with KDE
ax1 = fig.add_subplot(gs[0, :2])
ax1.hist(ov, bins=60, color=PURPLE, alpha=0.65, edgecolor='white', linewidth=0.4,
         density=True, label='Frequency')
ov_kde = ov[ov < ov.quantile(0.99)]
kde_x = np.linspace(ov_kde.min(), ov_kde.max(), 300)
kde = stats.gaussian_kde(ov_kde)
ax1.plot(kde_x, kde(kde_x), color=CORAL, linewidth=2.2, label='KDE')
ax1.axvline(mean_val,   color=AMBER, linewidth=2, linestyle='--', label=f'Mean ₹{mean_val:,.0f}')
ax1.axvline(median_val, color=TEAL,  linewidth=2, linestyle='-',  label=f'Median ₹{median_val:,.0f}')
ax1.set_title('Order Value Distribution — Right Skewed', fontweight='bold')
ax1.set_xlabel('Order Value (₹)'); ax1.set_ylabel('Density')
ax1.legend(fontsize=9); ax1.set_xlim(0, ov.quantile(0.97))
ax1.text(0.72, 0.82, f'Skewness: {skew_val:.2f}', transform=ax1.transAxes,
         fontsize=10, color=CORAL,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#FAECE7', alpha=0.8))

# Panel 2: Box plot
ax2 = fig.add_subplot(gs[0, 2])
bp = ax2.boxplot(ov, patch_artist=True, vert=True, widths=0.5,
                 boxprops=dict(facecolor=LGRAY, color=PURPLE),
                 whiskerprops=dict(color=PURPLE),
                 capprops=dict(color=PURPLE),
                 flierprops=dict(marker='.', markerfacecolor=CORAL, alpha=0.4, markersize=4),
                 medianprops=dict(color=TEAL, linewidth=2.5))
ax2.set_title('Box Plot with Outliers', fontweight='bold')
ax2.set_ylabel('Order Value (₹)')
ax2.set_xticklabels(['All Orders'])
ax2.text(1.12, median_val, f'Median\n₹{median_val:,.0f}',
         va='center', fontsize=8, color=TEAL)

# Panel 3: Category comparison
ax3 = fig.add_subplot(gs[1, :2])
cat_order = df.groupby('category')['order_value_inr'].median().sort_values(ascending=False).index
colors_cat = [PURPLE, BLUE, TEAL, AMBER, CORAL, DGRAY]
medians    = [df[df['category']==c]['order_value_inr'].median() for c in cat_order]
means_c    = [df[df['category']==c]['order_value_inr'].mean()   for c in cat_order]
x_pos = np.arange(len(cat_order))
bars = ax3.bar(x_pos, medians, color=colors_cat, alpha=0.8, width=0.4, label='Median')
ax3.scatter(x_pos, means_c, color='black', zorder=5, s=60, marker='D', label='Mean')
ax3.set_title('Median vs Mean Order Value by Category', fontweight='bold')
ax3.set_xticks(x_pos); ax3.set_xticklabels(cat_order, rotation=15, ha='right')
ax3.set_ylabel('Order Value (₹)')
ax3.legend()
for bar, m in zip(bars, medians):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
             f'₹{m:,.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Panel 4: Percentile curve
ax4 = fig.add_subplot(gs[1, 2])
pcts = np.arange(1, 100)
pct_vals = np.percentile(ov, pcts)
ax4.plot(pcts, pct_vals, color=BLUE, linewidth=2)
ax4.fill_between(pcts, pct_vals, alpha=0.12, color=BLUE)
for p, label, col in [(25,'Q1',TEAL),(50,'Median',AMBER),(75,'Q3',CORAL),(90,'P90',PURPLE)]:
    v = np.percentile(ov, p)
    ax4.axhline(v, linestyle='--', linewidth=1, color=col, alpha=0.7)
    ax4.text(p+1, v+300, f'{label}\n₹{v:,.0f}', fontsize=7, color=col)
ax4.set_title('Percentile Curve', fontweight='bold')
ax4.set_xlabel('Percentile'); ax4.set_ylabel('Order Value (₹)')

# Panel 5: Age group violin
ax5 = fig.add_subplot(gs[2, :2])
age_order = ['18-25','26-35','36-45','46-55','55+']
data_by_age = [df[df['age_group']==a]['order_value_inr'].values for a in age_order]
parts = ax5.violinplot(data_by_age, positions=range(len(age_order)), showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors_cat[i]); pc.set_alpha(0.6)
parts['cmedians'].set_color(DGRAY); parts['cmedians'].set_linewidth(2)
ax5.set_title('Order Value Distribution by Age Group (Violin Plot)', fontweight='bold')
ax5.set_xticks(range(len(age_order))); ax5.set_xticklabels(age_order)
ax5.set_ylabel('Order Value (₹)')

# Panel 6: Summary stats table
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')
stats_data = [
    ['Metric', 'Value'],
    ['Count',       f'{len(df):,}'],
    ['Mean',        f'₹{mean_val:,.0f}'],
    ['Median',      f'₹{median_val:,.0f}'],
    ['Std Dev',     f'₹{std_val:,.0f}'],
    ['Min',         f'₹{ov.min():,.0f}'],
    ['Max',         f'₹{ov.max():,.0f}'],
    ['IQR',         f'₹{iqr:,.0f}'],
    ['Skewness',    f'{skew_val:.3f}'],
    ['Kurtosis',    f'{kurt_val:.3f}'],
    ['# Outliers',  f'{len(outliers)} ({len(outliers)/len(df)*100:.1f}%)'],
]
tbl = ax6.table(cellText=stats_data[1:], colLabels=stats_data[0],
                loc='center', cellLoc='left')
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
tbl.auto_set_column_width([0,1])
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor('#E0E0E0')
    if row == 0:
        cell.set_facecolor(PURPLE); cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#F8F6FF')
    else:
        cell.set_facecolor('white')
ax6.set_title('Summary Statistics', fontweight='bold', pad=10)

plt.savefig('/home/claude/descriptive_stats/descriptive_stats_dashboard.png',
            dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
print("\n✓ Dashboard saved: descriptive_stats_dashboard.png")
