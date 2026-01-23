import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
game_df = pd.read_csv("../data/game_data.csv")

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

game_df['Date'] = game_df['Date'].apply(parse_game_date)
game_df = game_df.dropna(subset=['Date'])

# Aggregate across all users
agg_df = (
    game_df.groupby('Date')[['xG conceded', 'Goals against']]
    .mean()
    .sort_index()
)

# Smooth trend (7-game rolling mean)
agg_df_rolling = agg_df.rolling(window=7, min_periods=1).mean()

# Plot
sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(9, 5))

plt.plot(agg_df_rolling.index, agg_df_rolling['xG conceded'], label='xG Conceded', linewidth=2.2, color='steelblue')
plt.plot(agg_df_rolling.index, agg_df_rolling['Goals against'], label='Goals Against', linewidth=2.2, color='orange')

plt.title("In-Game Performance Development Over Time", pad=15)
plt.xlabel("Date")
plt.ylabel("7-Game Rolling Mean")
plt.legend()
plt.tight_layout()
plt.show()
