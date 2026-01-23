import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_1samp

# Load data
progress_df = pd.read_csv("../data/sensearena_progress.csv")
summary_df = pd.read_csv("../data/sensearena_summary.csv")

merged_df = progress_df.merge(summary_df, on='User', how='left')
merged_df = merged_df.dropna(subset=['GapMed', 'DrillEntropy'])

# Descriptive stats
mean_normslope = merged_df['NormSlope'].mean()
print(f"Normalized slope: {mean_normslope:.4f}")

t_stat, p_val = ttest_1samp(merged_df['NormSlope'], 0)
print(f"T-test (NormSlope > 0): t={t_stat:.3f}, p={p_val:.4f}\n")

# Correlations
corr_gap, p_gap = pearsonr(merged_df['GapMed'], merged_df['NormSlope'])
corr_top3, p_top3 = pearsonr(merged_df['DrillEntropy'], merged_df['NormSlope'])

r2_gap = corr_gap**2
r2_top3 = corr_top3**2

print(f"GapMed correlation: r={corr_gap:.3f}, R²={r2_gap:.3f}, p={p_gap:.4f}")
print(f"DrillEntropy correlation: r={corr_top3:.3f}, R²={r2_top3:.3f}, p={p_top3:.4f}")

# Visualization
""" plt.figure(figsize=(8, 5))
sns.histplot(merged_df['NormSlope'], bins=30, kde=True, color='skyblue')
mean_val = merged_df['NormSlope'].mean()
plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.3f}')
plt.title("Distribution of Normalized Improvement Slopes")
plt.xlabel("Normalized Slope (Improvement Rate)")
plt.ylabel("Count")
plt.legend()
plt.show() """

plt.figure(figsize=(10, 5))
sns.scatterplot(data=merged_df, x='DrillEntropy', y='NormSlope', alpha=0.7)
sns.regplot(data=merged_df, x='DrillEntropy', y='NormSlope', scatter=False, color='red')
plt.title('Relationship between Top 3 Drill Share and Improvement Rate')
plt.xlabel('DrillEntropy')
plt.ylabel('Normalized Improvement Slope')
plt.show()
