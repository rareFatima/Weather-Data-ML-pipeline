# Run ingestion, processing, feature preparation, and model stages in order.
from pathlib import Path
import subprocess
import sys


def main():
    project_folder = Path(__file__).resolve().parent

    stages = [
        ("Collect raw data", "src/ingest.py"),
        ("Clean and validate data", "src/process.py"),
        ("Create features", "src/features.py"),
        ("Prepare model inputs", "src/train.py"),
        ("Generate and save predictions", "src/predict.py")
    ]

    for name, script in stages:
        print(f"\nStarting: {name}", flush=True)

        try:
            subprocess.run(
                [sys.executable, script],
                cwd=project_folder,
                check=True
            )
        except subprocess.CalledProcessError:
            print(f"Pipeline stopped at: {name}", flush=True)
            raise

        print(f"Finished: {name}", flush=True)

    print("\nPipeline completed successfully!", flush=True)


if __name__ == "__main__":
    main()
