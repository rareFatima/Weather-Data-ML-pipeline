# Riyadh Weather Data Pipeline

An end-to-end data engineering project that collects hourly Riyadh
weather data from Open-Meteo, validates it, creates features, and
uses pretrained Chronos-2 to forecast temperature and rainfall.

## Pipeline

1. ingest.py collects weather data and uploads the original JSON.
2. process.py validates and cleans the data, then uploads Parquet.
3. features.py creates station-specific lag and time features.
4. train.py prepares historical context, future inputs, and actual
   values, and loads Chronos-2.
5. predict.py generates and saves forecasts.

main.py runs these stages in order. Evaluation and uploading the
forecast results are currently separate steps.

## Data

- Location: Riyadh
- Period: 2025-10-30 to 2026-09-28
- Total hourly records: 8016
- Historical context: 7848 hours
- Test period: 168 hours, from September 22 to September 28, 2026
- Timezone: UTC
- Targets: temperature_2m and rain
- Other weather variables: humidity, pressure, and wind speed

The test period is historical data held out from the model inputs.

## Features

- Temperature and rain lags of 1 and 24 hours
- Hour-of-day sine and cosine features

Lags are calculated separately for each station. Future inputs
contain timestamps and known time features only. Actual test
weather values and target-derived test lags are excluded.

## Setup

Run these commands from the project directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Install CPU PyTorch first to avoid downloading NVIDIA libraries.

Set up Google Cloud Application Default Credentials:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project cloud-based-weather-data
```

Your account needs access to the project and storage bucket.

## Run the Pipeline

```bash
source .venv/bin/activate
python main.py
```

Chronos-2 weights are downloaded on the first model run.
The pipeline stops if a stage fails.

## Evaluate the Forecasts

```bash
python src/evaluate_8016_checked.py
```

Results are saved in:

```text
data/predictions/experiment-8016-features/
```

Files:

- chronos_8016.csv: forecasts and quantiles
- actual_8016.csv: actual test values
- metrics_8016.csv: evaluation metrics

Upload the results:

```bash
gcloud storage cp data/predictions/experiment-8016-features/chronos_8016.csv data/predictions/experiment-8016-features/actual_8016.csv data/predictions/experiment-8016-features/metrics_8016.csv gs://riyadh-weather-ml-data/predictions/experiment-8016-features/
```

## Run the Dashboard

```bash
python -m streamlit run dashboard.py
```

The dashboard displays actual versus predicted weather and
evaluation metrics. Cloud-backed files require bucket access.

## Cloud Storage

Bucket: riyadh-weather-ml-data

- Raw JSON: raw/
- Clean data: processed/experiment-8016/weather_clean.parquet
- Feature-based forecasts and metrics:
  predictions/experiment-8016-features/

Features and model input files are currently saved locally.

## Results

Temperature results for the 168-hour test period:

| Inputs | Mean absolute error | Root mean squared error |
|---|---:|---:|
| Before new features | 0.882°C | 1.048°C |
| With new features | 0.833°C | 1.000°C |

The improvement applies to this test period and does not establish
performance across all seasons.

No rain occurred during the test period. Rainfall metrics therefore
do not establish the model's ability to predict rainfall events.
Values displayed as 0.000 are rounded, not necessarily exactly zero.
Raw model rainfall forecasts can include small negative values.

## Notes

- Chronos-2 is used without fine-tuning.
- The current final pipeline is specific to the 8016-row Riyadh experiment.
- Some dataset paths and expected row counts are fixed in the code.
- Local data is excluded from Git; code is shared through GitHub.
