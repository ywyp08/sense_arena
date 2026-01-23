import pandas as pd
from datetime import timedelta
from scipy import stats

# ============================================================
# CONFIG
# ============================================================

GAME_CSV_PATH = "../data/game_data_prepared.csv"
VR_CSV_PATH   = "../data/sa_data_prepared.csv"

USER_COL      = "User"
GAME_DATE_COL = "Date"
VR_DATE_COL   = "Date"
DRILL_COL     = "DrillName"

METRIC_COL    = "xG conceded"

LAG_DAYS_FOR_DRILLS = 3
MIN_GAMES_WITH_DRILL = 100
TOP_X_DRILLS = 15


# ============================================================
# HELPER: SIGNIFICANCE STARS
# ============================================================

def pstars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


# ============================================================
# LOAD DATA
# ============================================================

game_df = pd.read_csv(GAME_CSV_PATH, parse_dates=[GAME_DATE_COL])
sa_df   = pd.read_csv(VR_CSV_PATH,   parse_dates=[VR_DATE_COL])

# Keep only common users
common_users = set(game_df[USER_COL]).intersection(sa_df[USER_COL])
game_df = game_df[game_df[USER_COL].isin(common_users)].copy()
sa_df   = sa_df[sa_df[USER_COL].isin(common_users)].copy()


# ============================================================
# PREP: GROUP VR DATA BY USER
# ============================================================

vr_by_user = {
    user: grp[[VR_DATE_COL, DRILL_COL]].sort_values(VR_DATE_COL).reset_index(drop=True)
    for user, grp in sa_df.groupby(USER_COL)
}

# Ensure VR dates are datetime
for user, df in vr_by_user.items():
    df[VR_DATE_COL] = pd.to_datetime(df[VR_DATE_COL])
    vr_by_user[user] = df

game_df = game_df.reset_index(drop=False).rename(columns={"index": "GameIndex"})

# Sanity check metric column
if METRIC_COL not in game_df.columns:
    raise ValueError(f"Metric column '{METRIC_COL}' not found in game_df")


# ============================================================
# BUILD GAME–DRILL TABLE
# ============================================================

records = []

print(f"\nCollecting drills in the last {LAG_DAYS_FOR_DRILLS} day(s) before each game...")

for _, row in game_df.iterrows():
    user = row[USER_COL]
    gdate = pd.to_datetime(row[GAME_DATE_COL])
    game_idx = row["GameIndex"]

    vr_user = vr_by_user.get(user)
    if vr_user is None or vr_user.empty:
        continue

    start_date = gdate - timedelta(days=LAG_DAYS_FOR_DRILLS)
    mask = (vr_user[VR_DATE_COL] >= start_date) & (vr_user[VR_DATE_COL] <= gdate)
    drills_window = vr_user.loc[mask, DRILL_COL]

    if drills_window.empty:
        continue

    # Use unique drills per game
    for drill in drills_window.dropna().unique():
        records.append({
            "GameIndex": game_idx,
            USER_COL: user,
            "DrillName": drill
        })

drill_game_df = pd.DataFrame(records)
print(f"Game–drill pairs found: {len(drill_game_df)}")
print(f"Unique drills in window: {drill_game_df['DrillName'].nunique()}")


# ============================================================
# CALCULATE DAYS BEFORE GAME
# ============================================================

def get_days_before_game(row):
    user = row[USER_COL]
    drill = row["DrillName"]
    gdate = pd.to_datetime(game_df.loc[game_df["GameIndex"] == row["GameIndex"], GAME_DATE_COL].values[0])
    vr_user = vr_by_user.get(user)
    if vr_user is None:
        return pd.NA

    vr_user[VR_DATE_COL] = pd.to_datetime(vr_user[VR_DATE_COL])
    mask = (vr_user[DRILL_COL] == drill) & (vr_user[VR_DATE_COL] <= gdate) & \
           (vr_user[VR_DATE_COL] >= gdate - timedelta(days=LAG_DAYS_FOR_DRILLS))
    vr_dates = vr_user.loc[mask, VR_DATE_COL]
    if vr_dates.empty:
        return pd.NA

    return (gdate - vr_dates.max()).days

drill_game_df["DaysBeforeGame"] = drill_game_df.apply(get_days_before_game, axis=1)


# ============================================================
# DRILL FREQUENCIES (descriptive)
# ============================================================

drill_counts = (
    drill_game_df
    .groupby("DrillName")
    .agg(
        games_with_drill=("GameIndex", "nunique"),
        goalies_using_drill=(USER_COL, "nunique"),
        AvgDaysBeforeGame=("DaysBeforeGame", "mean")
    )
    .reset_index()
    .sort_values("games_with_drill", ascending=False)
)

print(f"\nTop {TOP_X_DRILLS} most common drills (by number of games where used in the window):")
print(drill_counts.head(TOP_X_DRILLS).round(2).to_string(index=False))


# ============================================================
# DRILL-LEVEL PERFORMANCE ANALYSIS (xG only)
# ============================================================

results = []

print("\nRunning drill-level xG analysis...")

for drill, sub in drill_game_df.groupby("DrillName"):
    games_with_idx = sub["GameIndex"].unique()
    n_with = len(games_with_idx)

    if n_with < MIN_GAMES_WITH_DRILL:
        continue  # skip rarely used drills

    users_with_drill = sub[USER_COL].unique()
    games_user_subset = game_df[game_df[USER_COL].isin(users_with_drill)]

    games_with = games_user_subset[games_user_subset["GameIndex"].isin(games_with_idx)]
    games_without = games_user_subset[~games_user_subset["GameIndex"].isin(games_with_idx)]

    xg_with = games_with[METRIC_COL]
    xg_without = games_without[METRIC_COL]

    if len(xg_without) == 0:
        continue

    mean_with = xg_with.mean()
    mean_without = xg_without.mean()
    diff = mean_with - mean_without

    tstat, pval = stats.ttest_ind(xg_with, xg_without, equal_var=False, nan_policy="omit")

    avg_days_before = sub["DaysBeforeGame"].mean()

    results.append({
        "DrillName": drill,
        "games_with": len(xg_with),
        "games_without": len(xg_without),
        "goalies_using_drill": len(users_with_drill),
        "mean_xg_with": mean_with,
        "mean_xg_without": mean_without,
        "diff": diff,
        "tstat": tstat,
        "pval": pval,
        "pstars": pstars(pval),
        "AvgDaysBeforeGame": avg_days_before
    })

results_df = pd.DataFrame(results)

if results_df.empty:
    print(f"\nNo drills passed the MIN_GAMES_WITH_DRILL threshold ({MIN_GAMES_WITH_DRILL}). Try lowering it.")
else:
    results_df = results_df.sort_values(by=["diff", "games_with"], ascending=[True, False]).reset_index(drop=True)
    print(f"\nDrills analyzed (n >= {MIN_GAMES_WITH_DRILL} games): {len(results_df)}")

    print(f"\nTop {TOP_X_DRILLS} drills by xG impact (sorted by diff_with_minus_without):")
    cols_to_show = [
        "DrillName", "games_with", "games_without", "goalies_using_drill",
        "mean_xg_with", "mean_xg_without", "diff", "pval", "pstars",
        "AvgDaysBeforeGame"
    ]
    print(results_df[cols_to_show].round(3).head(TOP_X_DRILLS).to_string(index=False))




# Drill amount analysis

# ============================================================
# 1️⃣ COUNT NUMBER OF UNIQUE DRILLS PER GAME
# ============================================================

# Count unique drills in window per game
drills_per_game = (
    drill_game_df.groupby("GameIndex")
    .agg(NumDrills=("DrillName", "nunique"))
    .reset_index()
)

# Merge with game metrics
game_dose_df = game_df.merge(drills_per_game, on="GameIndex", how="left")
game_dose_df["NumDrills"] = game_dose_df["NumDrills"].fillna(0)  # no drills → 0

# ============================================================
# 2️⃣ CALCULATE PERFORMANCE BY NUMBER OF DRILLS (DIFF FROM 0 DRILLS)
# ============================================================

# Baseline: mean metric for games with 0 drills
baseline_mean = game_dose_df.loc[game_dose_df["NumDrills"] == 0, METRIC_COL].mean()
baseline_std  = game_dose_df.loc[game_dose_df["NumDrills"] == 0, METRIC_COL].std()
baseline_n    = game_dose_df.loc[game_dose_df["NumDrills"] == 0, METRIC_COL].count()
baseline_sem  = baseline_std / baseline_n**0.5

# Group by number of drills
dose_summary = (
    game_dose_df.groupby("NumDrills")[METRIC_COL]
    .agg(["mean", "count", "std"])
    .reset_index()
)

# Compute difference from baseline
dose_summary["diff_from_0"] = dose_summary["mean"] - baseline_mean

# 95% CI for the difference (using independent samples approximation)
dose_summary["sem"] = dose_summary["std"] / dose_summary["count"] ** 0.5
dose_summary["ci_lower"] = dose_summary["diff_from_0"] - 1.96 * dose_summary["sem"]
dose_summary["ci_upper"] = dose_summary["diff_from_0"] + 1.96 * dose_summary["sem"]

print("\nPerformance diff vs 0 drills:")
print(dose_summary[["NumDrills", "count", "mean", "diff_from_0", "ci_lower", "ci_upper"]].round(3).to_string(index=False))

# ============================================================
# 3️⃣ PLOT DIFF VS NUMBER OF DRILLS
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.1)
plt.figure(figsize=(8,5))

plt.errorbar(
    dose_summary["NumDrills"],
    dose_summary["diff_from_0"],
    yerr=1.96*dose_summary["sem"],  # 95% CI
    fmt='o-', capsize=4, color="tab:green"
)

plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.title(f"Effect of Number of Pre-Game Drills on {METRIC_COL} (diff vs 0 drills)")
plt.xlabel("Number of Unique Drills in Pre-Game Window")
plt.ylabel(f"Change in {METRIC_COL} relative to 0 drills (lower = better)")
plt.xticks(range(0, int(dose_summary["NumDrills"].max()) + 1))
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 4️⃣ SMOOTHED DOSE-RESPONSE (LOWESS)
# ============================================================

import statsmodels.api as sm

# Prepare data
x = dose_summary["NumDrills"]
y = dose_summary["diff_from_0"]

# LOWESS smoothing (fraction controls smoothing window, e.g., 0.3 = 30% of data)
lowess_smoothed = sm.nonparametric.lowess(
    endog=y,
    exog=x,
    frac=0.3,  # adjust between 0.2–0.5 for more/less smooth
    return_sorted=True
)

smooth_x = lowess_smoothed[:, 0]
smooth_y = lowess_smoothed[:, 1]

# ============================================================
# 5️⃣ PLOT SMOOTHED DOSE-RESPONSE
# ============================================================

sns.set(style="whitegrid", font_scale=1.1)
plt.figure(figsize=(8,5))

# Raw points with error bars
plt.errorbar(
    dose_summary["NumDrills"],
    dose_summary["diff_from_0"],
    yerr=1.96*dose_summary["sem"],
    fmt='o', capsize=4, color="tab:blue", alpha=0.5, label="Observed"
)

# Smoothed curve
plt.plot(smooth_x, smooth_y, color="tab:red", linewidth=2, label="LOWESS smoothed")

plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.title(f"Smoothed Effect of Number of Pre-Game Drills on {METRIC_COL}")
plt.xlabel("Number of Unique Drills in Pre-Game Window")
plt.ylabel(f"Change in {METRIC_COL} relative to 0 drills (lower = better)")
plt.xticks(range(0, int(dose_summary["NumDrills"].max()) + 1))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()





#IDK
MIN_GAMES_FOR_CONFIDENCE = 30

# Identify points with enough games
valid_mask = dose_summary["count"] >= MIN_GAMES_FOR_CONFIDENCE
x_valid = dose_summary["NumDrills"][valid_mask]
y_valid = dose_summary["diff_from_0"][valid_mask]

# LOWESS only on reliable points
lowess_smoothed = sm.nonparametric.lowess(
    endog=y_valid,
    exog=x_valid,
    frac=0.3,
    return_sorted=True
)
smooth_x = lowess_smoothed[:, 0]
smooth_y = lowess_smoothed[:, 1]

# Plot
plt.figure(figsize=(8,5))
plt.errorbar(
    dose_summary["NumDrills"],
    dose_summary["diff_from_0"],
    yerr=1.96*dose_summary["sem"],
    fmt='o', capsize=4, color="tab:blue", alpha=0.5, label="Observed"
)
plt.plot(smooth_x, smooth_y, color="tab:red", linewidth=2, label="LOWESS smoothed")

# Shade low-sample regions
low_sample_mask = dose_summary["count"] < MIN_GAMES_FOR_CONFIDENCE
plt.fill_between(
    dose_summary["NumDrills"][low_sample_mask],
    dose_summary["diff_from_0"][low_sample_mask]-1.96*dose_summary["sem"][low_sample_mask],
    dose_summary["diff_from_0"][low_sample_mask]+1.96*dose_summary["sem"][low_sample_mask],
    color='gray', alpha=0.3, label="Low sample (<30 games)"
)

# Highlight minimal effective dose
min_effective_idx = smooth_y.argmin()  # minimum point in smoothed curve
plt.scatter(smooth_x[min_effective_idx], smooth_y[min_effective_idx],
            color="green", s=100, zorder=5, label="Minimal effective dose")

plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.title(f"Smoothed Effect of Number of Pre-Game Drills on {METRIC_COL}")
plt.xlabel("Number of Unique Drills in Pre-Game Window")
plt.ylabel(f"Change in {METRIC_COL} relative to 0 drills (lower = better)")
plt.xticks(range(0, int(dose_summary["NumDrills"].max()) + 1))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

