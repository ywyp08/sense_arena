import numpy as np
import pandas as pd
from datetime import timedelta
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIG
# ============================================================

GAME_CSV_PATH = "../data/game_data_prepared.csv"
VR_CSV_PATH   = "../data/sa_data_prepared.csv"

PERFORMANCE_COLS = ["xG conceded", "Goals against"]

LAG_DAYS_LIST = [0, 1, 2, 3, 4, 5, 6, 7]


# ============================================================
# LOAD DATA
# ============================================================

game_df = pd.read_csv(GAME_CSV_PATH, parse_dates=["Date"])
sa_df   = pd.read_csv(VR_CSV_PATH,   parse_dates=["Date"])

common_users = set(game_df["User"]).intersection(sa_df["User"])
game_df = game_df[game_df["User"].isin(common_users)].copy()
sa_df   = sa_df[sa_df["User"].isin(common_users)].copy()


# ============================================================
# FAST LOOKUP FOR VR SESSIONS
# ============================================================

vr_by_user = (
    sa_df.groupby("User")["Date"]
    .apply(lambda s: s.sort_values().to_numpy())
    .to_dict()
)

def had_vr(goalie, gdate, lag_days):
    dates = vr_by_user.get(goalie)
    if dates is None or len(dates) == 0:
        return False
    start_np = np.datetime64(gdate - timedelta(days=lag_days+1))
    end_np   = np.datetime64(gdate - timedelta(days=lag_days))
    return bool(((dates >= start_np) & (dates <= end_np)).any())

def had_vr_exact(goalie, gdate, lag_days):
    dates = vr_by_user.get(goalie)
    if dates is None or len(dates) == 0:
        return False
    target = np.datetime64(gdate - timedelta(days=lag_days))
    return bool((dates == target).any())


def had_vr_window(goalie, gdate, lag_days, tol=1):
    """
    Returns True if VR was done within [lag_days - tol, lag_days + tol] before the game.
    tol=0 → exact
    tol=1 → ±1 day window
    """
    dates = vr_by_user.get(goalie)
    if dates is None or len(dates) == 0:
        return False
    start = np.datetime64(gdate - timedelta(days=lag_days+tol))
    end   = np.datetime64(gdate - timedelta(days=lag_days-tol))
    return bool(((dates >= start) & (dates <= end)).any())



# ============================================================
# ADD LAG FLAGS
# ============================================================

for lag in LAG_DAYS_LIST:
    col = f"vr_training_{lag}d"
    game_df[col] = game_df.apply(
        lambda row: had_vr_exact(row["User"], row["Date"], lag),
        axis=1
    )


# ============================================================
# HELPER: SIGNIFICANCE STARS
# ============================================================

def pstars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


# ============================================================
# ANALYSIS + CLEAN TABLES
# ============================================================

lag_results = []

for lag in LAG_DAYS_LIST:
    lag_col = f"vr_training_{lag}d"

    for metric in PERFORMANCE_COLS:
        with_vr = game_df.loc[game_df[lag_col], metric]
        without_vr = game_df.loc[~game_df[lag_col], metric]

        tstat, pval = stats.ttest_ind(
            with_vr, without_vr,
            equal_var=False, nan_policy="omit"
        )

        lag_results.append({
            "lag_days": lag,
            "metric": metric,
            "mean_with": with_vr.mean(),
            "mean_without": without_vr.mean(),
            "diff": with_vr.mean() - without_vr.mean(),
            "pval": pval,
            "pstars": pstars(pval),
            "n_with": len(with_vr),
            "n_without": len(without_vr)
        })

lag_df = pd.DataFrame(lag_results)

# Split tables
xg_df = lag_df[lag_df["metric"] == "xG conceded"].copy()
ga_df = lag_df[lag_df["metric"] == "Goals against"].copy()

print("\n\n=== xG Conceded — Lag Results (1–7 days) ===")
print(xg_df[["lag_days","mean_with","mean_without","diff","pval","pstars","n_with","n_without"]]
      .round(3).to_string(index=False))

print("\n\n=== Goals Against — Lag Results (1–7 days) ===")
print(ga_df[["lag_days","mean_with","mean_without","diff","pval","pstars","n_with","n_without"]]
      .round(3).to_string(index=False))


# ============================================================
# PLOT DIFFERENCE VS LAG
# ============================================================

sns.set(style="whitegrid", font_scale=1.1)
plt.figure(figsize=(8,5))

sns.lineplot(data=lag_df,
             x="lag_days",
             y="diff",
             hue="metric",
             marker="o")

plt.axhline(0, linestyle="--")
plt.title("Performance Difference (With VR – Without VR)\nby VR Training Window Length (1–7 days)")
plt.xlabel("Days before the Match")
plt.ylabel("Difference in Performance")
plt.tight_layout()
plt.show()
