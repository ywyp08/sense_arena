import pandas as pd
from itertools import combinations
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
MIN_GAMES_FOR_COMBO = 10   # only consider combos with >= this many games
MAX_COMBO_SIZE = 3         # 1 = single drill, 2 = pairs, 3 = triplets

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
sa_df   = pd.read_csv(VR_CSV_PATH, parse_dates=[VR_DATE_COL])

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

# ============================================================
# BUILD GAME–DRILL TABLE
# ============================================================

records = []
for _, row in game_df.iterrows():
    user = row[USER_COL]
    gdate = pd.to_datetime(row[GAME_DATE_COL])
    game_idx = row["GameIndex"]

    vr_user = vr_by_user.get(user)
    if vr_user is None or vr_user.empty:
        continue

    start_date = gdate - pd.Timedelta(days=LAG_DAYS_FOR_DRILLS)
    mask = (vr_user[VR_DATE_COL] >= start_date) & (vr_user[VR_DATE_COL] <= gdate)
    drills_window = vr_user.loc[mask, DRILL_COL]

    if drills_window.empty:
        continue

    for drill in drills_window.dropna().unique():
        records.append({
            "GameIndex": game_idx,
            USER_COL: user,
            "DrillName": drill
        })

drill_game_df = pd.DataFrame(records)

# ============================================================
# BUILD GAME × DRILL BINARY MATRIX
# ============================================================

drill_game_matrix = pd.get_dummies(drill_game_df[['GameIndex','DrillName']], columns=['DrillName'])
drill_game_matrix = drill_game_matrix.groupby('GameIndex').max()  # one row per game

# Merge with game performance metric
game_drill_df = game_df[['GameIndex', METRIC_COL]].merge(drill_game_matrix, on='GameIndex', how='left')
game_drill_df.fillna(0, inplace=True)

# List of drill columns
drills = [col for col in game_drill_df.columns if col.startswith('DrillName_')]

# ============================================================
# FIND EFFECTIVE DRILL COMBINATIONS
# ============================================================

results = []

for combo_size in range(1, MAX_COMBO_SIZE+1):
    for combo in combinations(drills, combo_size):
        mask_with_combo = game_drill_df[list(combo)].all(axis=1)
        xg_with = game_drill_df.loc[mask_with_combo, METRIC_COL]
        xg_without = game_drill_df.loc[~mask_with_combo, METRIC_COL]

        if len(xg_with) < MIN_GAMES_FOR_COMBO or len(xg_without) == 0:
            continue

        mean_with = xg_with.mean()
        mean_without = xg_without.mean()
        diff = mean_with - mean_without
        tstat, pval = stats.ttest_ind(xg_with, xg_without, equal_var=False, nan_policy='omit')

        results.append({
            'DrillCombo': combo,
            'n_games_with': len(xg_with),
            'n_games_without': len(xg_without),
            'mean_xg_with': mean_with,
            'mean_xg_without': mean_without,
            'diff': diff,
            'tstat': tstat,
            'pval': pval,
            'pstars': pstars(pval)
        })

# ============================================================
# SHOW TOP COMBOS
# ============================================================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('diff').reset_index(drop=True)  # negative diff = better

print("\nTop 20 drill combinations by in-game improvement (negative diff = better):")
print(results_df.head(20).to_string(index=False))
