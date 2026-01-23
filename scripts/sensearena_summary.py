import numpy as np
import pandas as pd
from scipy.stats import entropy

# Load data
df = pd.read_csv('../data/sensearena_data.csv')

# ---- SESSION-LEVEL DATA ----
df['CreatedAt'] = pd.to_datetime(df['CreatedAt'], errors='coerce')
df['SessionDate'] = df['CreatedAt'].dt.date

# DaysGap as days (numeric) with NaN for same-day zeroes
df['DaysGap'] = (
    df.groupby('User')['CreatedAt']
      .diff()
      .dt.days
)
df['DaysGap'] = df['DaysGap'].replace(0, np.nan)

# ---- BASIC GAP STATS ----
gap_stats = df.groupby('User')['DaysGap'].agg(
    GapAvg='mean',
    GapMed='median',
    GapStd='std',
    GapMax='max'
)

# ---- VARIETY ----
def shannon_entropy(x):
    counts = x.value_counts(normalize=True)
    return entropy(counts)

def normalized_entropy(x):
    counts = x.value_counts(normalize=True)
    return entropy(counts) / np.log(len(counts))

variety_stats = df.groupby('User')['DrillId'].agg(
    TotalDrills='count',
    UniqueDrills='nunique',
    DrillEntropy=shannon_entropy,
    NormDrillEntropy=normalized_entropy
)

# Top-K Drill Share
def top_k_share(s, k):
    return s.sort_values(ascending=False).head(k).sum() / s.sum()

drill_counts = (
    df.groupby(['User', 'DrillId'])
      .size()
      .groupby('User')
)

topk_shares = pd.DataFrame({
    f'Top{k}Share': drill_counts.apply(lambda s, k=k: top_k_share(s, k))
    for k in range(1, 10)
})



# ---- SESSION STRUCTURE ----
session_stats = (
    df.groupby(['User', 'SessionDate'])
      .size()
      .groupby('User')
      .agg(
          AvgDrillsPerSession='mean',
          MaxDrillsPerSession='max'
      )
)

# ---- CONSISTENCY ----
active_days = df.groupby('User')['SessionDate'].nunique()
total_days = (df['SessionDate'].max() - df['SessionDate'].min()).days + 1
active_days_pct = (active_days / total_days * 100).rename('ActiveDaysPct')

df['Week'] = df['CreatedAt'].dt.isocalendar().week
active_weeks = df.groupby('User')['Week'].nunique()
total_weeks = df['Week'].nunique()
active_weeks_pct = (active_weeks / total_weeks * 100).rename('ActiveWeeksPct')

def longest_streak(dates):
    dates = sorted(set(dates))
    streak = max_streak = 1
    for i in range(1, len(dates)):
        if (dates[i] - dates[i-1]).days == 1:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 1
    return max_streak

streaks = df.groupby('User')['SessionDate'].apply(
    lambda x: longest_streak(pd.to_datetime(x))
).rename('LongestStreak')

# ---- SEASONAL SPLIT ----
season_gap = df.groupby(['User', 'InSeason'])['DaysGap'].mean().unstack().rename(
    columns={0: 'GapAvg_OffSeason', 1: 'GapAvg_InSeason'}
)

inseason_pct = (
    df.groupby('User')['InSeason']
      .mean()
      .mul(100)
      .rename('InSeasonPct')
)

# ---- COMBINE METRICS INTO SUMMARY ----
summary_df = (
    gap_stats
    .join(variety_stats)
    .join(topk_shares)
    .join(session_stats)
    .join(active_days_pct)
    .join(active_weeks_pct)
    .join(streaks)
    .join(season_gap)
    .join(inseason_pct)
)

# Reset index to make 'User' a column
summary_df = summary_df.reset_index()

# Compute FirstDrillDate per user
first_drill_date = df.groupby('User')['CreatedAt'].min().rename('FirstDrillDate')

# Insert FirstDrillDate right after User
summary_df.insert(
    loc=1,
    column='FirstDrillDate',
    value=pd.to_datetime(summary_df['User'].map(first_drill_date)).dt.date
)

# Round numeric columns for readability
summary_df = summary_df.round({
    'GapAvg': 2, 'GapMed': 2, 'GapStd': 2, 'GapMax': 2,
    'DrillEntropy': 2, 'NormDrillEntropy': 2,
    'Top1Share': 2, 'Top2Share': 2, 'Top3Share': 2,
    'Top4Share': 2, 'Top5Share': 2, 'Top6Share': 2,
    'AvgDrillsPerSession': 2, 'MaxDrillsPerSession': 0,
    'ActiveDaysPct': 1, 'ActiveWeeksPct': 1,
    'GapAvg_OffSeason': 2, 'GapAvg_InSeason': 2,
    'InSeasonPct': 1
})


# Save to CSV
summary_df.to_csv("../data/sensearena_summary.csv", index=False)


