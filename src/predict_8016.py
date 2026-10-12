from pathlib import Path
import pandas as pd
from chronos import Chronos2Pipeline

df = pd.read_csv(
    "data/processed/riyadh_8016.csv",
    parse_dates=["time"]
)
df = df.sort_values("time").reset_index(drop=True)
df["item_id"] = "riyadh"

context = df.iloc[:-168].copy()
test = df.iloc[-168:].copy()

print("Context rows:", len(context))
print("Test rows:", len(test))

print("Loading model...")
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cpu"
)

print("Generating predictions...")
predictions = pipeline.predict_df(
    context,
    timestamp_column="time",
    target=["temperature_2m", "rain"],
    prediction_length=168,
    context_length=len(context),
    batch_size=1,
    freq="h"
)

Path("data/predictions").mkdir(parents=True, exist_ok=True)

predictions.to_csv(
    "data/predictions/chronos_8016.csv",
    index=False
)
test.to_csv(
    "data/predictions/actual_8016.csv",
    index=False
)

print(predictions.head())
print("Predictions and actual values saved.")
