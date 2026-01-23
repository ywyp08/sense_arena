import pandas as pd
import numpy as np
from scipy.stats import linregress

# Load data
df = pd.read_csv("../data/sensearena_data.csv")

# Ensure datetime
df['CreatedAt'] = pd.to_datetime(df['CreatedAt'])

# Min sessions filter
min_sessions = 20

results = []

# Group by user+drill+settings
for (user, drill, settings), group in df.groupby(['User', 'DrillId', 'Settings']):
    if len(group) >= min_sessions:
        group = group.sort_values('CreatedAt')
        x = np.arange(len(group))
        y = group['Score'].values

        score_range = y.max() - y.min()
        if score_range == 0:
            continue

        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        norm_slope = slope / score_range

        results.append({
            'User': user,
            'DrillId': drill,
            'Settings': settings,
            'Sessions': len(group),
            'RawSlope': slope,
            'NormSlope': norm_slope,
            'R2': r_value**2,
            'MeanScore': y.mean()
        })

results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv("../data/sensearena_progress.csv", index=False)

