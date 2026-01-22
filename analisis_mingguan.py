# analyze_data_lari.py
import pandas as pd
import numpy as np
import json
import os
from scipy import stats
import matplotlib.pyplot as plt

CSV_PATH = "data_lari_saya.csv"   # sesuaikan jika nama file berbeda
OUTPUT_DIR = "outputs"
MIN_WEEKS_PER_SIDE = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) Load
df = pd.read_csv(CSV_PATH)
print("Columns:", df.columns.tolist())

# 2) Pick date column from common candidates and parse
date_candidates = ['start_date_local','start_date','date','Activity Date','Activity Date (Local)']
date_col = next((c for c in date_candidates if c in df.columns), None)
if date_col is None:
    raise SystemExit("No date column found among candidates. Columns: " + ", ".join(df.columns))
df['activity_date'] = pd.to_datetime(df[date_col], errors='coerce')
df = df.dropna(subset=['activity_date'])
print("Using date column:", date_col, "| rows after date parse:", len(df))

# 3) Distance: prefer distance_km if present, else infer unit from 'distance'
if 'distance_km' in df.columns:
    df['distance_km'] = pd.to_numeric(df['distance_km'], errors='coerce')
    print("Using 'distance_km' column (km).")
else:
    if 'distance' not in df.columns:
        raise SystemExit("No distance column found.")
    sample = pd.to_numeric(df['distance'].dropna().iloc[:10], errors='coerce')
    if (sample > 100).mean() > 0.5:
        df['distance_km'] = pd.to_numeric(df['distance'], errors='coerce') / 1000.0
        print("Inferred 'distance' in meters; converted to distance_km by dividing by 1000.")
    else:
        df['distance_km'] = pd.to_numeric(df['distance'], errors='coerce')
        print("Inferred 'distance' already in km; using as distance_km.")
df = df.dropna(subset=['distance_km'])
df = df[df['distance_km'] > 0]

# 4) Elapsed time: prefer moving_time, else elapsed_time or duration_min
if 'moving_time' in df.columns:
    df['elapsed_time_s'] = pd.to_numeric(df['moving_time'], errors='coerce')
    print("Using 'moving_time' as elapsed_time (seconds).")
elif 'elapsed_time' in df.columns:
    df['elapsed_time_s'] = pd.to_numeric(df['elapsed_time'], errors='coerce')
    print("Using 'elapsed_time' as elapsed_time (seconds).")
elif 'duration_min' in df.columns:
    df['elapsed_time_s'] = pd.to_numeric(df['duration_min'], errors='coerce') * 60.0
    print("Using 'duration_min' converted to seconds as elapsed_time.")
else:
    df['elapsed_time_s'] = np.nan
    print("No elapsed time column found; pace must be present.")

# 5) Pace: use pace_decimal if exists, else compute
if 'pace_decimal' in df.columns and df['pace_decimal'].notna().any():
    df['pace_min_per_km'] = pd.to_numeric(df['pace_decimal'], errors='coerce')
    df = df.dropna(subset=['pace_min_per_km'])
    print("Used existing 'pace_decimal' as pace (min/km).")
else:
    df = df.dropna(subset=['elapsed_time_s','distance_km'])
    df['pace_min_per_km'] = (df['elapsed_time_s'] / df['distance_km']) / 60.0
    print("Computed pace from elapsed_time_s and distance_km.")

# 6) (Optional) filter to Run type if column exists
if 'activity_type' in df.columns:
    df = df[df['activity_type'].str.contains('run', case=False, na=False)]
    print("Filtered to runs. Count:", len(df))

# 7) Aggregate weekly (ISO week)
df['year'] = df['activity_date'].dt.isocalendar().year
df['week'] = df['activity_date'].dt.isocalendar().week
weekly = df.groupby(['year','week']).agg(
    n_runs=('pace_min_per_km', 'count'),
    avg_pace=('pace_min_per_km', 'mean'),
    med_pace=('pace_min_per_km', 'median'),
    sd_pace=('pace_min_per_km', 'std'),
    total_km=('distance_km', 'sum')
).reset_index().sort_values(['year','week']).reset_index(drop=True)

n_weeks = len(weekly)
print(f"Computed weekly aggregation: {n_weeks} weeks found.")

if n_weeks < (MIN_WEEKS_PER_SIDE*2 + 1):
    raise SystemExit(f"Not enough weeks ({n_weeks}) for automatic split detection. Need at least {(MIN_WEEKS_PER_SIDE*2 + 1)} weeks.")

# 8) Find best split (automatic) via two-sample t-test maximizing |t|
best = {'split_idx': None, 'tstat_abs': -np.inf, 'pvalue': None}
for i in range(MIN_WEEKS_PER_SIDE, n_weeks - MIN_WEEKS_PER_SIDE):
    left = weekly.loc[:i-1, 'avg_pace'].dropna()
    right = weekly.loc[i:, 'avg_pace'].dropna()
    if len(left) < MIN_WEEKS_PER_SIDE or len(right) < MIN_WEEKS_PER_SIDE:
        continue
    tstat, pval = stats.ttest_ind(left, right, equal_var=False)
    if np.abs(tstat) > best['tstat_abs']:
        best.update({'split_idx': i, 'tstat_abs': np.abs(tstat), 'pvalue': pval})

if best['split_idx'] is None:
    raise SystemExit("Failed to find a valid split point.")

split_idx = best['split_idx']
weekly['phase'] = ['baseline' if idx < split_idx else 'post' for idx in range(len(weekly))]

baseline = weekly[weekly['phase']=='baseline']
post = weekly[weekly['phase']=='post']

summary = {
    'n_weeks_total': int(n_weeks),
    'split_index': int(split_idx),
    'split_week_year': int(weekly.loc[split_idx, 'year']),
    'split_week_week': int(weekly.loc[split_idx, 'week']),
    'baseline_mean_pace': float(baseline['avg_pace'].mean()),
    'post_mean_pace': float(post['avg_pace'].mean()),
    'baseline_total_km': float(baseline['total_km'].sum()),
    'post_total_km': float(post['total_km'].sum()),
    'tstat_abs': float(best['tstat_abs']),
    'pvalue': float(best['pvalue'])
}

# 9) Save outputs
os.makedirs(OUTPUT_DIR, exist_ok=True)
weekly.to_csv(os.path.join(OUTPUT_DIR, "weekly_summary.csv"), index=False)
with open(os.path.join(OUTPUT_DIR, "summary.json"), 'w') as f:
    json.dump(summary, f, indent=4)

# 10) Plots
plt.figure(figsize=(10,5))
plt.plot(weekly.index, weekly['avg_pace'], marker='o', label='Avg Pace (min/km)')
plt.axvline(x=split_idx, color='red', linestyle='--', label='Detected split (intervention start)')
plt.gca().invert_yaxis()
plt.xlabel('Week index (chronological)')
plt.ylabel('Average pace (min/km)')
plt.title('Weekly average pace with detected intervention split')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(OUTPUT_DIR, "weekly_pace.png"), bbox_inches='tight')
plt.close()

# Quick classification and correlation plot
pace_Q1 = df['pace_min_per_km'].quantile(0.25)
pace_Q3 = df['pace_min_per_km'].quantile(0.75)
def classify(p):
    if p <= pace_Q1: return 'speed'
    if p >= pace_Q3: return 'easy'
    return 'tempo'
df['workout_type_smart'] = df['pace_min_per_km'].apply(classify)
weekly2 = df.groupby(['year','week']).agg(
    avg_pace=('pace_min_per_km','mean'),
    speed_runs=('workout_type_smart', lambda x: (x=='speed').sum()),
    easy_runs=('workout_type_smart', lambda x: (x=='easy').sum())
).reset_index().sort_values(['year','week']).reset_index(drop=True)
weekly2['pace_delta'] = weekly2['avg_pace'].shift(-1) - weekly2['avg_pace']

plt.figure(figsize=(8,6))
x = weekly2['speed_runs'].fillna(0)
y = weekly2['pace_delta'].fillna(0)
plt.scatter(x, y, alpha=0.7, s=70)
if len(x) > 1:
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color='red', linewidth=2)
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Number of speed runs previous week')
plt.ylabel('Pace delta (next week - this week) [min/km]')
plt.title(f'Speed runs vs next-week pace delta (r={weekly2["speed_runs"].corr(weekly2["pace_delta"]):.3f})')
plt.savefig(os.path.join(OUTPUT_DIR, "speed_vs_pacedelta.png"), bbox_inches='tight')
plt.close()

print("Selesai. Output disimpan di folder:", OUTPUT_DIR)
print("Ringkasan:", json.dumps(summary, indent=2))
