import json
import logging
from pathlib import Path

import pandas as pd
import yaml
from google.cloud import storage


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

weather_columns = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "rain"
]


def fill_short_gaps(df):
    affected_rows = set()

    for column in weather_columns:
        missing = df[column].isna()
        groups = missing.ne(missing.shift()).cumsum()

        for _, indexes in df[missing].groupby(groups[missing]).groups.items():
            indexes = list(indexes)
            start = indexes[0]
            end = indexes[-1]

            if (
                len(indexes) <= 2
                and start > 0
                and end < len(df) - 1
                and pd.notna(df.loc[start - 1, column])
                and pd.notna(df.loc[end + 1, column])
            ):
                values = df.loc[
                    start - 1:end + 1, column
                ].interpolate(method="linear")

                df.loc[start:end, column] = values.loc[start:end]
                affected_rows.update(indexes)

    logging.info(f"Rows interpolated: {len(affected_rows)}")
    return df


def main():
    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    with open("data/raw/riyadh_2025-10-30_2026-09-28.json") as file:
        data = json.load(file)

    df = pd.DataFrame(data["hourly"])
    logging.info(f"Rows loaded: {len(df)}")

    # Check required fields
    required_columns = ["time"] + weather_columns
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    df = df[required_columns].copy()

    # The original dataset must use UTC
    if data.get("utc_offset_seconds") != 0:
        raise ValueError("Source timezone needs review")

    df["station"] = "riyadh"
    df["timestamp_utc"] = pd.to_datetime(
        df["time"], utc=True, errors="coerce"
    )

    invalid_times = df["timestamp_utc"].isna().sum()
    logging.info(f"Invalid timestamps removed: {invalid_times}")

    df = df.dropna(subset=["timestamp_utc"])
    df = df.drop(columns=["time"])
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(
        subset=["station", "timestamp_utc"]
    )
    logging.info(f"Duplicates removed: {before - len(df)}")

    if df.empty:
        raise ValueError("No valid rows remain")

    # Add rows for missing hours
    df = df.set_index("timestamp_utc")

    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="h"
    )

    if len(df.index.difference(full_range)) > 0:
        raise ValueError("Off-grid timestamps found")

    missing_hours = len(full_range.difference(df.index))
    logging.info(f"Missing hours found: {missing_hours}")

    df = df.reindex(full_range)
    df.index.name = "timestamp_utc"
    df["station"] = "riyadh"
    df = df.reset_index()

    # Convert weather values to numbers
    for column in weather_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].replace(
            [float("inf"), float("-inf")], float("nan")
        )

    # Check plausible ranges
    invalid_values = pd.DataFrame({
        "temperature_2m": df["temperature_2m"].notna()
        & ~df["temperature_2m"].between(-90, 60),

        "relative_humidity_2m": df["relative_humidity_2m"].notna()
        & ~df["relative_humidity_2m"].between(0, 100),

        "surface_pressure": df["surface_pressure"] <= 0,
        "wind_speed_10m": df["wind_speed_10m"] < 0,
        "rain": df["rain"] < 0
    })

    logging.info(
        f"Rows with invalid ranges: {invalid_values.any(axis=1).sum()}"
    )

    for column in weather_columns:
        df.loc[invalid_values[column], column] = float("nan")

    logging.info(
        f"Rows with missing weather values: "
        f"{df[weather_columns].isna().any(axis=1).sum()}"
    )

    # Fill only complete gaps of 1 or 2 hours
    df = fill_short_gaps(df)

    # Drop longer gaps and gaps without valid neighbours
    before = len(df)
    df = df.dropna(subset=weather_columns)
    logging.info(f"Rows dropped: {before - len(df)}")

    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    if df.empty:
        raise ValueError("No clean rows remain")

    # Save Parquet with station and timestamp_utc
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    parquet_path = "data/processed/riyadh_8016.parquet"
    df.to_parquet(parquet_path, index=False, engine="pyarrow")

    # Keep CSV compatible with the existing prediction script
    csv_df = df.rename(columns={"timestamp_utc": "time"})
    csv_df = csv_df[["time"] + weather_columns]
    csv_df.to_csv("data/processed/riyadh_8016.csv", index=False)

    logging.info(f"Final clean rows: {len(df)}")
    logging.info("Saved local CSV and Parquet")

    # Upload to a separate experiment folder
    client = storage.Client()
    bucket = client.bucket(config["bucket_name"])
    cloud_path = "processed/experiment-8016/weather_clean.parquet"

    blob = bucket.blob(cloud_path)
    blob.upload_from_filename(
        parquet_path,
        content_type="application/octet-stream"
    )

    logging.info(
        f"Uploaded: gs://{config['bucket_name']}/{cloud_path}"
    )


if __name__ == "__main__":
    main()
