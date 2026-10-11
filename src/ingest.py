# Fetch weather data and store the original JSON response.
from pathlib import Path
from google.cloud import storage
import requests
import yaml


def ingest():
    project_folder = Path(__file__).resolve().parent.parent
    config_path = project_folder / "config.yaml"

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    raw_folder = project_folder / "data" / "raw"
    raw_folder.mkdir(parents=True, exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(config["bucket_name"])

    for station in config["stations"]:
        url = "https://archive-api.open-meteo.com/v1/archive"

        params = {
            "latitude": station["latitude"],
            "longitude": station["longitude"],
            "start_date": config["start_date"],
            "end_date": config["end_date"],
            "hourly": ",".join(config["hourly_variables"]),
            "timezone": config["timezone"],
        }

        print("Location:", station["name"])
        print("Date range:", config["start_date"], "to", config["end_date"])

        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
        except requests.Timeout:
            print("Request timed out.")
            raise
        except requests.RequestException as error:
            print("Request failed:", error)
            raise

        city = station["name"].lower()
        start_date = config["start_date"]
        end_date = config["end_date"]

        filename = f"{city}_{start_date}_{end_date}.json"
        file_path = raw_folder / filename

        with open(file_path, "wb") as file:
            file.write(response.content)

        print("Data saved:", file_path)

        blob = bucket.blob(f"raw/{filename}")
        blob.upload_from_filename(
            str(file_path),
            content_type="application/json"
        )

        print("Uploaded:", f"gs://{config['bucket_name']}/raw/{filename}")


if __name__ == "__main__":
    ingest()
