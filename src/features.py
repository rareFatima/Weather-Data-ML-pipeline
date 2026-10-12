# Prepare features and model inputs.

import pandas as pd
import numpy as np
from pathlib import Path

# Read the cleaned dataset
input_path = Path("data/processed/riyadh_8016.parquet")
output_path = Path("data/features/riyadh_8016_features.parquet")

df = pd.read_parquet(input_path)

# Ensure UTC timestamps and sort by station and time
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
df = df.sort_values(["station", "timestamp_utc"]).reset_index(drop=True)

# Verify hourly continuity before creating lag features
hour_gaps = (
    df.groupby("station")["timestamp_utc"]
    .diff()
    .dropna()
    .ne(pd.Timedelta(hours=1))
    .sum()
)
if hour_gaps:
    raise ValueError(f"Found {hour_gaps} non-hourly intervals.")

# Previous-hour and previous-24-hour features, separately per station
df["temperature_lag_1h"] = df.groupby("station")["temperature_2m"].shift(1)
df["temperature_lag_24h"] = df.groupby("station")["temperature_2m"].shift(24)
df["rain_lag_1h"] = df.groupby("station")["rain"].shift(1)
df["rain_lag_24h"] = df.groupby("station")["rain"].shift(24)

# Daily cycle features based on the UTC hour
hour = df["timestamp_utc"].dt.hour
df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

# Save to a new file; keep the clean source unchanged
df.to_parquet(output_path, index=False)

print("Output file:", output_path)
print("Rows:", len(df))
print("\nMissing values per column:")
print(df.isna().sum())
