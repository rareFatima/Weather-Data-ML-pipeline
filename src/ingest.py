# Fetch weather data and store the original JSON response.
import logging
import time
from pathlib import Path

import requests
import yaml
from google.cloud import storage


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)


def fetch_weather(url, params):
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        logging.info("API request: attempt %s/%s", attempt, max_attempts)

        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response

        except requests.HTTPError as error:
            status = error.response.status_code

            # Retry temporary server errors and rate limits.
            if status != 429 and not 500 <= status < 600:
                logging.error("API rejected the request: %s", error)
                raise

            if attempt == max_attempts:
                logging.error("API request failed after all attempts")
                raise

            logging.warning("Temporary API error: HTTP %s", status)

        except (requests.Timeout, requests.ConnectionError) as error:
            if attempt == max_attempts:
                logging.error("API request failed after all attempts: %s", error)
                raise

            logging.warning("Temporary connection problem: %s", error)

        # Wait 2 seconds, then 4 seconds before retrying.
        wait_seconds = 2 ** attempt
        logging.info("Retrying in %s seconds", wait_seconds)
        time.sleep(wait_seconds)


def ingest():
    logging.info("Ingestion started")

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

        logging.info("Location: %s", station["name"])
        logging.info(
            "Date range: %s to %s",
            config["start_date"],
            config["end_date"]
        )

        response = fetch_weather(url, params)

        # Check the response before saving its original bytes.
        data = response.json()
        rows = len(data["hourly"]["time"])
        logging.info("Hourly rows received: %s", rows)

        city = station["name"].lower()
        start_date = config["start_date"]
        end_date = config["end_date"]

        filename = f"{city}_{start_date}_{end_date}.json"
        file_path = raw_folder / filename

        with open(file_path, "wb") as file:
            file.write(response.content)

        logging.info("Data saved: %s", file_path)

        blob = bucket.blob(f"raw/{filename}")
        blob.upload_from_filename(
            str(file_path),
            content_type="application/json"
        )

        logging.info(
            "Uploaded: gs://%s/raw/%s",
            config["bucket_name"],
            filename
        )

    logging.info("Ingestion completed")


if __name__ == "__main__":
    ingest()
