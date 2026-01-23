import pandas as pd
from scipy import stats


# ============================================================
# CONFIG
# ============================================================

GAME_CSV_PATH = "../data/game_data_prepared.csv"
VR_CSV_PATH   = "../data/sa_data_prepared.csv"

USER_COL      = "User"
GAME_DATE_COL = "Date"
VR_DATE_COL   = "Date"

METRIC_COL    = "xG conceded"

# Experience thresholds to test (total VR sessions)
VR_THRESHOLDS = [5, 10, 20, 30, 50, 80]

OUTPUT_CSV = "../data/vr_experience_threshold_results.csv"


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

print("Loading prepared data...")
game_df = pd.read_csv(GAME_CSV_PATH, parse_dates=[GAME_DATE_COL])
vr_df   = pd.read_csv(VR_CSV_PATH,   parse_dates=[VR_DATE_COL])

print(f"Games loaded: {len(game_df)}")
print(f"VR sessions loaded: {len(vr_df)}")

# Keep only common users
common_users = set(game_df[USER_COL]).intersection(vr_df[USER_COL])
game_df = game_df[game_df[USER_COL].isin(common_users)].copy()
vr_df   = vr_df[vr_df[USER_COL].isin(common_users)].copy()

print(f"Common users: {len(common_users)}")
print(f"Games after filtering: {len(game_df)}")
print(f"VR sessions after filtering: {len(vr_df)}")

# Sanity check metric column
if METRIC_COL not in game_df.columns:
    raise ValueError(f"Metric column '{METRIC_COL}' not found in game_df")


# ============================================================
# BUILD CUMULATIVE VR EXPERIENCE PER USER (NO merge_asof)
# ============================================================

# Sort for consistency (not strictly needed for the counting logic)
game_df = game_df.sort_values([USER_COL, GAME_DATE_COL]).reset_index(drop=True)
vr_df   = vr_df.sort_values([USER_COL, VR_DATE_COL]).reset_index(drop=True)

# Build a dict: user -> sorted numpy array of their VR session dates
vr_by_user_dates = {
    user: grp[VR_DATE_COL].sort_values().to_numpy()
    for user, grp in vr_df.groupby(USER_COL)
}

# Initialize column
game_df["cum_vr_experience"] = 0

for idx, row in game_df.iterrows():
    user = row[USER_COL]
    gdate = row[GAME_DATE_COL]

    dates = vr_by_user_dates.get(user)
    if dates is None or len(dates) == 0:
        # no VR for this user before this game
        continue

    # Count how many VR sessions are <= game date
    # searchsorted with side='right' gives index of first value > gdate
    # which is exactly "number of values <= gdate"
    count = dates.searchsorted(gdate.to_datetime64(), side="right")
    game_df.at[idx, "cum_vr_experience"] = int(count)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

results = []

print("\nRunning VR experience threshold analysis...\n")

for threshold in VR_THRESHOLDS:
    post = game_df.loc[game_df["cum_vr_experience"] >= threshold, METRIC_COL]
    pre  = game_df.loc[game_df["cum_vr_experience"] < threshold, METRIC_COL]

    if len(post) == 0 or len(pre) == 0:
        print(f"Threshold {threshold}: skipped (not enough games pre/post)")
        continue

    mean_post = post.mean()
    mean_pre  = pre.mean()
    diff = mean_post - mean_pre  # post - pre

    tstat, pval = stats.ttest_ind(
        post, pre,
        equal_var=False,
        nan_policy="omit"
    )

    results.append({
        "threshold": threshold,
        "games_pre": len(pre),
        "games_post": len(post),
        "mean_pre": mean_pre,
        "mean_post": mean_post,
        "diff_post_minus_pre": diff,
        "tstat": tstat,
        "pval": pval,
        "pstars": pstars(pval),
    })

results_df = pd.DataFrame(results)

print("=== VR Experience Threshold Effects on xG Conceded ===\n")
if not results_df.empty:
    print(
        results_df[[
            "threshold","games_pre","games_post",
            "mean_pre","mean_post","diff_post_minus_pre",
            "pval","pstars"
        ]].round(3).to_string(index=False)
    )
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")
else:
    print("No thresholds produced valid pre/post splits. Try lowering thresholds or checking data.")
