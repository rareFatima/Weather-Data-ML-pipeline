# Validate, clean, and save processed weather data.
import json
import pandas as pd
from pathlib import Path

with open("data/raw/riyadh_2025-10-30_2026-09-28.json") as file:
    data = json.load(file)

df = pd.DataFrame(data["hourly"])
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time")

print("Rows:", len(df))
print("Missing values:")
print(df.isna().sum())
print("Duplicate timestamps:", df["time"].duplicated().sum())
print(df.head())

Path("data/processed").mkdir(parents=True, exist_ok=True)

df.to_csv("data/processed/riyadh_8016.csv", index=False)
print("Saved: data/processed/riyadh_8016.csv")
