from pathlib import Path

import pandas as pd
from chronos import Chronos2Pipeline


# 1. Load cleaned data
df = pd.read_parquet(
    "data/processed/weather_clean.parquet"
)

# Rename timestamp column for Chronos
df = df.rename(
    columns={"timestamp_utc": "time"}
)

# Convert to datetime
df["time"] = pd.to_datetime(
    df["time"],
    utc=True
)

# Chronos expects timezone-naive datetime64[ns]
df["time"] = df["time"].dt.tz_localize(None)

# Sort chronologically
df = df.sort_values(
    "time"
).reset_index(drop=True)



# 2. Select required columns
columns = [
    "time",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "rain",
]

df = df[columns].copy()

# Series identifier
df["item_id"] = "riyadh"



# 3. Chronological holdout
# Last 168 hours = 7 days

context = df.iloc[:-168].copy()
test = df.iloc[-168:].copy()

print("Total rows:", len(df))
print("Context rows:", len(context))
print("Test rows:", len(test))
print("Test starts:", test["time"].min())
print("Test ends:", test["time"].max())

print("Time dtype:", df["time"].dtype)


# 4. Load pretrained Chronos-2 model

print("Loading model...")

pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cpu"
)


# 5. Generate predictions

print("Generating predictions...")

predictions = pipeline.predict_df(
    context,
    id_column="item_id",
    timestamp_column="time",
    target=["temperature_2m", "rain"],
    prediction_length=168,
    context_length=len(context),
    batch_size=1,
    freq="h"
)


# 6. Save predictions

Path(
    "data/predictions"
).mkdir(
    parents=True,
    exist_ok=True
)

predictions.to_csv(
    "data/predictions/chronos_2160.csv",
    index=False
)

test.to_csv(
    "data/predictions/actual_2160.csv",
    index=False
)



# 7. Preview results

print(predictions.head())

print(
    "Predictions and actual values saved."
)
