from pathlib import Path

import numpy as np
import pandas as pd

PREDICTIONS_DIR = Path("data/predictions")


def main():
    # 1. Load the two files
    predictions = pd.read_csv(PREDICTIONS_DIR / "chronos_8016.csv")
    actual = pd.read_csv(PREDICTIONS_DIR / "actual_8016.csv")

    # 2. Use the same time format in both files
    predictions["time"] = pd.to_datetime(predictions["time"], utc=True)
    actual["time"] = pd.to_datetime(actual["time"], utc=True)

    rows = []

    # 3. One target at a time, using the target_name column
    for target in predictions["target_name"].unique():
        predicted = predictions[predictions["target_name"] == target]
        predicted = predicted[["time", "predictions"]]

        # 4. Match predicted and actual values for the same hour
        merged = predicted.merge(
            actual[["time", target]], on="time", how="inner"
        )

        if len(merged) != len(predicted):
            raise ValueError(f"Some hours have no actual value for {target}")

        # 5. Error for each hour, then the two metrics
        error = merged["predictions"] - merged[target]
        mae = error.abs().mean()
        rmse = np.sqrt((error ** 2).mean())

        rows.append({
            "target": target,
            "hours": len(merged),
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            # Hours where the actual value is not zero (important for rain)
            "nonzero_actual_hours": int((merged[target] != 0).sum()),
        })

    # 6. Print and save the summary table
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    output_path = PREDICTIONS_DIR / "metrics_8016.csv"
    table.to_csv(output_path, index=False)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
