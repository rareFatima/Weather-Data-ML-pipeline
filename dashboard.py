from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from google.cloud import storage


BUCKET_NAME = "riyadh-weather-ml-data"
FOLDER = "predictions/experiment-8016-features"


@st.cache_data(ttl=300)
def load_data():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    files = []

    for name in ["chronos_8016.csv", "actual_8016.csv", "metrics_8016.csv"]:
        blob = bucket.blob(f"{FOLDER}/{name}")
        content = blob.download_as_bytes()
        data = pd.read_csv(BytesIO(content))
        files.append(data)

    return files[0], files[1], files[2]


st.set_page_config(page_title="Riyadh Weather", layout="wide")

st.title("Riyadh Weather Dashboard")
st.write("Dataset: 8,016 hourly records")
st.write("Test period: September 22–28, 2026")
st.write("Forecast length: 168 hours")
st.header("Clean Data Overview")

bucket = storage.Client().bucket("riyadh-weather-ml-data")
blob = bucket.blob(
    "processed/experiment-8016/weather_clean.parquet"
)

clean_data = pd.read_parquet(BytesIO(blob.download_as_bytes()))

st.write("Location: Riyadh")
st.write("Total records:", len(clean_data))
st.write("Start:", str(clean_data["timestamp_utc"].min()))
st.write("End:", str(clean_data["timestamp_utc"].max()))

st.dataframe(clean_data.head(), hide_index=True)

try:
    predictions, actual, metrics = load_data()
except Exception as error:
    st.error(f"Could not load the files: {error}")
    st.stop()

predictions["time"] = pd.to_datetime(predictions["time"], utc=True)
actual["time"] = pd.to_datetime(actual["time"], utc=True)

for target in ["temperature_2m", "rain"]:
    if target == "temperature_2m":
        title = "Temperature"
        unit = "°C"
    else:
        title = "Rainfall"
        unit = "mm"

    st.header(title)

    predicted = predictions[
        predictions["target_name"] == target
    ][["item_id", "time", "predictions"]]

    data = predicted.merge(
        actual[["item_id", "time", target]],
        on=["item_id", "time"],
        how="outer",
        validate="one_to_one",
        indicator=True
    )

    if len(data) != 168 or not data["_merge"].eq("both").all():
        st.error(f"Prediction and actual timestamps do not match for {title}.")
        st.stop()

    if data[["predictions", target]].isna().any().any():
        st.error(f"Missing values found for {title}.")
        st.stop()

    data = data.sort_values("time")

    chart = go.Figure()

    chart.add_trace(go.Scatter(
        x=data["time"],
        y=data[target],
        name="Actual",
        mode="lines"
    ))

    chart.add_trace(go.Scatter(
        x=data["time"],
        y=data["predictions"],
        name="Predicted",
        mode="lines",
        line={"dash": "dash"}
    ))

    chart.update_layout(
        xaxis_title="Time (UTC)",
        yaxis_title=f"{title} ({unit})"
    )

    st.plotly_chart(chart)

    target_metrics = metrics[metrics["target"] == target]

    if len(target_metrics) != 1:
        st.error(f"Missing or duplicate metrics for {title}.")
        st.stop()

    result = target_metrics.iloc[0]

    col1, col2 = st.columns(2)
    col1.metric("MAE", f"{result['MAE']:.3f} {unit}")
    col2.metric("RMSE", f"{result['RMSE']:.3f} {unit}")

    if target == "rain":
        rainy_hours = int((data["rain"] > 0).sum())
        st.write("Rainy hours:", rainy_hours)

        if rainy_hours == 0:
            st.warning(
                "No rain occurred during the test period. "
                "We need a rainy period to evaluate rainfall forecasting."
            )

st.caption(
    "Predicted lines show median forecasts. "
    "Errors are rounded to three decimal places."
)
