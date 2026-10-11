import json
from pathlib import Path
import pandas as pd

weather_columns = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "rain"
]

with open("data/raw/riyadh_2025-10-30_2026-09-28.json") as file:
    data = json.load(file)

df = pd.DataFrame(data["hourly"])

# 1. Check required columns
required_columns = ["time"] + weather_columns
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]
if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

df = df[required_columns].copy()
print("Rows loaded:", len(df))

# 2. Confirm source timezone and parse timestamps
if data.get("utc_offset_seconds") != 0:
    raise ValueError("Source timezone needs review before using UTC")

df["time"] = pd.to_datetime(
    df["time"], utc=True, errors="coerce"
)

invalid_times = df["time"].isna().sum()
print("Invalid timestamps:", invalid_times)
if invalid_times > 0:
    raise ValueError("Invalid timestamps found")

# 3. Sort and check duplicates
df = df.sort_values("time").reset_index(drop=True)

duplicates = df["time"].duplicated().sum()
print("Duplicate timestamps:", duplicates)
if duplicates > 0:
    raise ValueError("Duplicate timestamps found")

# 4. Check hourly continuity
if df.empty:
    raise ValueError("Dataset is empty")

expected_times = pd.date_range(
    start=df["time"].min(),
    end=df["time"].max(),
    freq="h"
)

actual_times = pd.DatetimeIndex(df["time"])
missing_hours = len(expected_times.difference(actual_times))
off_grid_times = len(actual_times.difference(expected_times))

print("Missing hours:", missing_hours)
print("Off-grid timestamps:", off_grid_times)

if missing_hours > 0 or off_grid_times > 0:
    raise ValueError("Hourly continuity needs review")

# 5. Convert weather values to numbers and check missing values
for column in weather_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

print("Missing or non-numeric values:")
print(df[weather_columns].isna().sum())

if df[weather_columns].isna().any().any():
    raise ValueError("Missing or non-numeric values found")

# 6. Check plausible ranges
invalid_values = pd.DataFrame({
    "temperature_2m": ~df["temperature_2m"].between(-90, 60),
    "relative_humidity_2m": ~df["relative_humidity_2m"].between(0, 100),
    "surface_pressure": df["surface_pressure"] <= 0,
    "wind_speed_10m": df["wind_speed_10m"] < 0,
    "rain": df["rain"] < 0
})

# Also reject infinity
for column in weather_columns:
    invalid_values[column] = (
        invalid_values[column]
        | df[column].isin([float("inf"), float("-inf")])
    )

print("Invalid values per column:")
print(invalid_values.sum())

if invalid_values.any().any():
    raise ValueError("Invalid weather values found")

# 7. Save only after all checks pass
Path("data/processed").mkdir(parents=True, exist_ok=True)
df.to_csv("data/processed/riyadh_8016.csv", index=False)

print("Validation passed")
print("Final rows:", len(df))
print("Saved: data/processed/riyadh_8016.csv")
