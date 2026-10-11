import json
import logging
from io import BytesIO

import pandas as pd
import yaml
from google.cloud import storage


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


REQUIRED_COLUMNS = [
    "time",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "rain",
]

WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "rain",
]


def load_config():
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)


def load_raw_data(bucket_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Read only the 2160-row dataset
    blob = bucket.blob(
        "raw/riyadh_2026-07-01_2026-09-28.json"
    )

    logging.info(f"Reading {blob.name}")

    raw_data = json.loads(blob.download_as_text())

    if "hourly" not in raw_data:
        raise ValueError("'hourly' not found in raw data")

    return pd.DataFrame(raw_data["hourly"])


def interpolate_short_gaps(df, columns):
    """
    Interpolate only complete gaps of 1 or 2 hours.
    Gaps longer than 2 hours remain missing and will be dropped.
    """

    interpolated_rows = set()

    for column in columns:
        missing = df[column].isna()

        # Give each consecutive missing/non-missing block an ID
        groups = missing.ne(missing.shift()).cumsum()

        for _, indexes in df[missing].groupby(groups[missing]).groups.items():
            indexes = list(indexes)

            # Only interpolate gaps of 1 or 2 rows
            if len(indexes) <= 2:
                start = indexes[0]
                end = indexes[-1]

                # Need a valid value before and after the gap
                if (
                    start > 0
                    and end < len(df) - 1
                    and pd.notna(df.loc[start - 1, column])
                    and pd.notna(df.loc[end + 1, column])
                ):
                    values = df.loc[
                        start - 1:end + 1, column
                    ].interpolate(method="linear")

                    df.loc[start:end, column] = values.loc[start:end]

                    interpolated_rows.update(indexes)

    return df, len(interpolated_rows)


def clean_data(df):
    # Add station
    df["station"] = "riyadh"

    # 1. Validate required columns
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # 2. Standardize timestamp to UTC
    df["timestamp_utc"] = pd.to_datetime(
        df["time"],
        utc=True,
        errors="coerce"
    )

    invalid_timestamps = df["timestamp_utc"].isna().sum()

    logging.info(
        f"Invalid timestamps: {invalid_timestamps}"
    )

    # Remove rows with invalid timestamps
    df = df.dropna(
        subset=["timestamp_utc"]
    )

    # Remove original time column
    df = df.drop(
        columns=["time"]
    )

    # 3. Sort chronologically
    df = df.sort_values(
        ["station", "timestamp_utc"]
    ).reset_index(drop=True)

    # 4. Remove duplicates
    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=["station", "timestamp_utc"]
    )

    duplicates_removed = (
        before_duplicates - len(df)
    )

    logging.info(
        f"Duplicates removed: {duplicates_removed}"
    )

    # Sort again after duplicate removal
    df = df.sort_values(
        ["station", "timestamp_utc"]
    ).reset_index(drop=True)

    # 5. Check hourly continuity
    df = df.set_index("timestamp_utc")

    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="h",
        tz="UTC"
    )

    missing_hours = len(
        full_range.difference(df.index)
    )

    logging.info(
        f"Missing hourly timestamps found: {missing_hours}"
    )

    # Create rows for completely missing hours
    df = df.reindex(full_range)

    df.index.name = "timestamp_utc"

    # We only have Riyadh in this dataset
    df["station"] = "riyadh"

    df = df.reset_index()

    # 6. Validate plausible value ranges
    invalid_values = (
        (
            df["relative_humidity_2m"].notna()
            & ~df["relative_humidity_2m"].between(0, 100)
        )
        |
        (
            df["wind_speed_10m"].notna()
            & (df["wind_speed_10m"] < 0)
        )
        |
        (
            df["rain"].notna()
            & (df["rain"] < 0)
        )
        |
        (
            df["surface_pressure"].notna()
            & (df["surface_pressure"] <= 0)
        )
        |
        (
            df["temperature_2m"].notna()
            & ~df["temperature_2m"].between(-90, 60)
        )
    )

    invalid_rows = invalid_values.sum()

    logging.info(
        f"Rows with invalid values: {invalid_rows}"
    )

    # Turn invalid weather values into missing values
    # so they go through the same missing-value policy.
    df.loc[
        invalid_values,
        WEATHER_COLUMNS
    ] = pd.NA

    # 7. Handle missing values
    df, interpolated_rows = interpolate_short_gaps(
        df,
        WEATHER_COLUMNS
    )

    logging.info(
        f"Rows interpolated: {interpolated_rows}"
    )

    # Anything still missing belongs to a longer gap
    # or cannot safely be interpolated.
    before_drop = len(df)

    df = df.dropna(
        subset=WEATHER_COLUMNS
    )

    dropped_rows = (
        before_drop - len(df)
    )

    logging.info(
        f"Rows dropped after missing-value handling: {dropped_rows}"
    )

    # 8. Final sorting
    df = df.sort_values(
        ["station", "timestamp_utc"]
    ).reset_index(drop=True)

    return df


def save_processed_data(df, bucket_name):
    buffer = BytesIO()

    df.to_parquet(
        buffer,
        index=False,
        engine="pyarrow"
    )

    buffer.seek(0)

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blob = bucket.blob(
        "processed/weather_clean.parquet"
    )

    blob.upload_from_file(
        buffer,
        content_type="application/octet-stream"
    )

    logging.info(
        "Saved processed/weather_clean.parquet to the bucket"
    )


def main():
    config = load_config()

    bucket_name = config["bucket_name"]

    # Load raw data
    df = load_raw_data(bucket_name)

    logging.info(
        f"Rows loaded: {len(df)}"
    )

    # Clean and validate data
    clean_df = clean_data(df)

    logging.info(
        f"Final clean rows: {len(clean_df)}"
    )

    # Preview cleaned data
    print(clean_df.head())
    print(clean_df.tail())

    # Save clean Parquet to Google Cloud
    save_processed_data(
        clean_df,
        bucket_name
    )


if __name__ == "__main__":
    main()
