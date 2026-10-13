from pathlib import Path

import pandas as pd

from train import PREDICTION_LENGTH, TARGETS, load_model, prepare_inputs

# New folder: results in data/predictions/ stay unchanged
OUTPUT_DIR = Path("data/predictions/experiment-8016-features")


def to_naive_time(df):
    """Store time as plain UTC datetime without the timezone label."""
    df = df.copy()
    df["time"] = (
        pd.to_datetime(df["time"], utc=True)
        .dt.tz_localize(None)
        .astype("datetime64[ns]")
    )
    return df


def main():
    # 1. Inputs prepared by train.py
    context, future, actual = prepare_inputs()

    context = to_naive_time(context)
    future = to_naive_time(future)
    actual = to_naive_time(actual)

    print("Context rows:", len(context))
    print("Future rows:", len(future))

    # 2. Model prepared by train.py
    print("Loading model...")
    pipeline = load_model()

    # 3. Predict the next 168 hours
    print("Generating predictions...")
    predictions = pipeline.predict_df(
        context,
        future_df=future,
        id_column="item_id",
        timestamp_column="time",
        target=TARGETS,
        prediction_length=PREDICTION_LENGTH,
        context_length=len(context),
        freq="h",
        batch_size=1,
    )

    # 4. Check: one prediction per hour for each target
    expected_rows = PREDICTION_LENGTH * len(TARGETS)
    if len(predictions) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} prediction rows, got {len(predictions)}"
        )

    # 5. Save predictions (with time and target name) and actual values
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(OUTPUT_DIR / "chronos_8016.csv", index=False)
    actual.to_csv(OUTPUT_DIR / "actual_8016.csv", index=False)

    print(predictions[["time", "target_name", "predictions"]].head())
    print("Saved:", OUTPUT_DIR / "chronos_8016.csv")
    print("Saved:", OUTPUT_DIR / "actual_8016.csv")


if __name__ == "__main__":
    main()