import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
summary_df = pd.read_csv('../data/sensearena_summary.csv')

# ---------------------------
# Regularity
plt.figure(figsize=(8,6))
sns.histplot(summary_df['GapMed'], bins=10, kde=True, color='tab:blue')
plt.title('Distribution of training regularity')
plt.xlabel('Median gap between sessions (days)')
plt.ylabel('Number of users')
plt.show()

# ---------------------------
# Top-K distribution (3x3 grid)
topk_cols = [f'Top{k}Share' for k in range(1, 10)]
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(topk_cols):
    sns.histplot(summary_df[col], bins=10, kde=True, ax=axes[i], color='tab:orange')
    axes[i].set_xlim(0,1)
    axes[i].set_title(col)
    axes[i].set_xlabel('Share')
    axes[i].set_ylabel('Users')

plt.tight_layout()
plt.show()

# ---------------------------
# Mean Top-K curve
mean_shares = summary_df[topk_cols].mean()
plt.figure(figsize=(7,4))
plt.plot(range(1,10), mean_shares, marker='o', color='tab:orange')
plt.xticks(range(1,10))
plt.xlabel('K (Top-K drills)')
plt.ylabel('Average drill share')
plt.title('Average drill concentration by Top-K')
plt.grid(True)
plt.show()

# ---------------------------
# Drill variety: entropy
fig, axes = plt.subplots(1,2, figsize=(14,6))

sns.histplot(summary_df['DrillEntropy'], bins=10, kde=True, color='tab:orange', ax=axes[0])
axes[0].set_title('Raw entropy')
axes[0].set_xlabel('Entropy')
axes[0].set_ylabel('Number of users')

sns.histplot(summary_df['NormDrillEntropy'], bins=10, kde=True, color='tab:orange', ax=axes[1])
axes[1].set_title('Normalized entropy')
axes[1].set_xlabel('Normalized entropy (0–1)')
axes[1].set_ylabel('Number of users')

plt.tight_layout()
plt.show()

