from pathlib import Path

import numpy as np
import pandas as pd

PREDICTIONS_DIR = Path("data/predictions")


def main():
    # Load the 8016 experiment files
    predictions = pd.read_csv(
        PREDICTIONS_DIR / "chronos_8016.csv"
    )
    actual = pd.read_csv(
        PREDICTIONS_DIR / "actual_8016.csv"
    )

    # Standardize timestamps
    predictions["time"] = pd.to_datetime(
        predictions["time"], utc=True
    )
    actual["time"] = pd.to_datetime(
        actual["time"], utc=True
    )

    rows = []

    # Evaluate each target separately
    for target in ["temperature_2m", "rain"]:
        predicted = predictions[
            predictions["target_name"] == target
        ][["time", "predictions"]].copy()

        merged = predicted.merge(
            actual[["time", target]],
            on="time",
            how="outer",
            validate="one_to_one",
            indicator=True
        )

        if not merged["_merge"].eq("both").all():
            raise ValueError(
                f"Unmatched timestamps for {target}"
            )

        if len(merged) != 168:
            raise ValueError(
                f"Expected 168 test hours for {target}"
            )

        merged = merged.sort_values("time")

        expected_times = pd.date_range(
            start="2026-09-22 00:00:00",
            periods=168,
            freq="h",
            tz="UTC"
        )

        if not pd.DatetimeIndex(merged["time"]).equals(expected_times):
            raise ValueError("Unexpected test period")

        # Check values before calculating errors
        values = merged[["predictions", target]]

        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(
                f"Missing or infinite values for {target}"
            )

        # Calculate MAE and RMSE
        error = merged["predictions"] - merged[target]
        mae = error.abs().mean()
        rmse = np.sqrt((error ** 2).mean())

        rows.append({
            "target": target,
            "hours": len(merged),
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "nonzero_actual_hours": int(
                (merged[target] != 0).sum()
            )
        })

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    output_path = PREDICTIONS_DIR / "metrics_8016.csv"
    table.to_csv(output_path, index=False)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
