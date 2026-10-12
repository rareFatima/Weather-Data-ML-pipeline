from pathlib import Path

import numpy as np
import pandas as pd
from chronos import Chronos2Pipeline


TARGETS = ["temperature_2m", "rain"]
PREDICTION_LENGTH = 168

WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "rain"
]

FEATURE_COLUMNS = [
    "temperature_lag_1h",
    "temperature_lag_24h",
    "rain_lag_1h",
    "rain_lag_24h",
    "hour_sin",
    "hour_cos"
]


def prepare_inputs():
    # Read the dataset prepared by features.py
    df = pd.read_parquet(
        "data/processed/riyadh_8016_features.parquet"
    )

    required_columns = (
        ["station", "timestamp_utc"]
        + WEATHER_COLUMNS
        + FEATURE_COLUMNS
    )

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    df = df[required_columns].copy()

    # Standardize and check timestamps
    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="raise"
    )

    if df[["station", "timestamp_utc"]].isna().any().any():
        raise ValueError("Missing station or timestamp")

    if df["station"].nunique() != 1:
        raise ValueError("Expected one station for this experiment")

    df = df.sort_values(
        "timestamp_utc"
    ).reset_index(drop=True)

    time_gaps = df["timestamp_utc"].diff().dropna()

    if not time_gaps.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Duplicate or irregular hourly timestamps")

    if len(df) != 8016:
        raise ValueError(f"Expected 8016 rows, found {len(df)}")

    weather_values = df[WEATHER_COLUMNS].to_numpy(dtype=float)

    if not np.isfinite(weather_values).all():
        raise ValueError("Missing or invalid weather values")

    # Use the names expected by our prediction workflow
    df = df.rename(columns={
        "station": "item_id",
        "timestamp_utc": "time"
    })

    # Historical inputs: 7848 hours
    context = df.iloc[:-PREDICTION_LENGTH].copy()

    # Holdout period: final 168 hours
    test = df.iloc[-PREDICTION_LENGTH:].copy()

    # Keep actual weather values separate for evaluation
    actual = test[
        ["item_id", "time"] + WEATHER_COLUMNS
    ].copy()

    # Future inputs contain only known time information.
    # Do not include actual weather values or target-derived lags.
    future = test[["item_id", "time"]].copy()

    hour = future["time"].dt.hour
    future["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    future["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    return context, future, actual


def load_model():
    # Load the pretrained model on CPU
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu"
    )

    return pipeline


def main():
    context, future, actual = prepare_inputs()

    output_dir = Path("data/model_inputs/experiment-8016")
    output_dir.mkdir(parents=True, exist_ok=True)

    context.to_parquet(
        output_dir / "context.parquet",
        index=False
    )

    future.to_parquet(
        output_dir / "future.parquet",
        index=False
    )

    actual.to_parquet(
        output_dir / "actual.parquet",
        index=False
    )

    print("Context rows:", len(context))
    print("Future rows:", len(future))
    print("Test starts:", actual["time"].min())
    print("Test ends:", actual["time"].max())
    print("Future columns:", future.columns.tolist())

    print("Loading model...")
    pipeline = load_model()

    print("Model ready:", type(pipeline).__name__)
    print("Saved model inputs:", output_dir)


if __name__ == "__main__":
    main()
