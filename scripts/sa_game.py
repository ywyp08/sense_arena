import pandas as pd
import numpy as np
from datetime import timedelta
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
game_df = pd.read_csv("../data/game_data.csv")
sa_df = pd.read_csv("../data/sensearena_data.csv")

# Parse dates
def parse_game_date(s: str, default_year: int = 2024) -> pd.Timestamp:
    s = str(s).strip()
    parts = s.split('/')
    if len(parts) == 3:
        day, month, yy = parts
        year = 2000 + int(yy)
        return pd.Timestamp(year, int(month), int(day))
    elif len(parts) == 2:
        day, month = parts
        return pd.Timestamp(default_year, int(month), int(day))
    else:
        return pd.to_datetime(s, errors='coerce')

game_df['Date'] = game_df['Date'].apply(lambda s: parse_game_date(s, default_year=2024))
sa_df['Date'] = pd.to_datetime(sa_df['CreatedAt'], errors='coerce')

sa_df["Date"] = pd.to_datetime(sa_df["Date"]).dt.date
game_df["Date"] = pd.to_datetime(game_df["Date"]).dt.date

sa_df['Date'] = sa_df['Date'].values.astype('datetime64[D]')
game_df['Date'] = game_df['Date'].values.astype('datetime64[D]')

print("Game parse failures:", game_df['Date'].isna().sum(), "/", len(game_df))
print("VR parse failures:", sa_df['Date'].isna().sum(), "/", len(sa_df))

# Drop rows with bad dates
game_df = game_df.dropna(subset=['Date'])
sa_df = sa_df.dropna(subset=['Date'])

# Filter to common goalies
common = set(game_df['User']).intersection(sa_df['User'])
game_df = game_df[game_df['User'].isin(common)]
sa_df = sa_df[sa_df['User'].isin(common)]

# Save the data
game_df.to_csv('../data/game_data_prepared.csv', index=False)
sa_df.to_csv('../data/sa_data_prepared.csv', index=False)

# Match VR training
def had_vr_training(goalie, gdate):
    drills = sa_df[sa_df['User'] == goalie]
    mask = (drills['Date'] >= gdate - timedelta(days=1)) & (drills['Date'] <= gdate)
    return mask.any()

game_df['vr_training'] = game_df.apply(lambda row: had_vr_training(row['User'], row['Date']), axis=1)

performance_cols = ['xG conceded', 'Goals against']
summary_vr = (
    game_df.groupby('vr_training')[performance_cols]
    .mean()
    .rename(index={True: 'With VR', False: 'Without VR'})
)

# Pre/post adoption
first_session = sa_df.groupby('User')['Date'].min().to_dict()
game_df['post_vr_period'] = game_df.apply(lambda row: row['Date'] >= first_session.get(row['User'], pd.Timestamp.max), axis=1)

summary_prepost = (
    game_df.groupby('post_vr_period')[performance_cols]
    .mean()
    .rename(index={True: 'After VR start', False: 'Before VR start'})
)

# Statistical testing
def paired_test(group1, group2, label1, label2, varname):
    """
    Performs a paired t-test (if data is paired) or fallback nonparametric test,
    prints and returns results.
    """
    tstat, pval = stats.ttest_ind(group1, group2, equal_var=False, nan_policy='omit')
    print(f"\nT-test ({varname}): {label1} vs {label2}")
    print(f"  t = {tstat:.3f}, p = {pval:.3f}")
    return tstat, pval

# Compare performance with vs without VR
print("\n=== Performance With vs Without VR ===")
print(summary_vr.round(3))
for col in performance_cols:
    g_with = game_df.loc[game_df['vr_training'], col]
    g_without = game_df.loc[~game_df['vr_training'], col]
    paired_test(g_with, g_without, "With VR", "Without VR", col)

# Compare before vs after VR adoption
print("\n\n=== Performance Before vs After VR Adoption ===")
print(summary_prepost.round(3))
for col in performance_cols:
    g_after = game_df.loc[game_df['post_vr_period'], col]
    g_before = game_df.loc[~game_df['post_vr_period'], col]
    paired_test(g_after, g_before, "After VR start", "Before VR start", col)
    
# Visualization

sns.set(style="whitegrid", font_scale=1.1)

# Prepare DataFrame for plotting
summary_vr_reset = summary_vr.reset_index()
summary_vr_reset.rename(columns={summary_vr_reset.columns[0]: 'VR Training'}, inplace=True)

# Melt for easier plotting
summary_vr_melted = summary_vr_reset.melt(
    id_vars='VR Training',
    var_name='Metric',
    value_name='Mean Value'
)

# Bar chart: metrics on x-axis, grouped by VR training
plt.figure(figsize=(7, 5))
sns.barplot(
    data=summary_vr_melted,
    x='Metric',
    y='Mean Value',
    hue='VR Training',
    palette=['#1f78b4', '#a6cee3']
)

plt.title("Game Performance — With vs Without VR training in previous 3 days")
plt.ylabel("Mean Value")
plt.xlabel("")
plt.legend(title="VR Training")
plt.tight_layout()
plt.show()