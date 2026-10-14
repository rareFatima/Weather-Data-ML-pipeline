from io import BytesIO
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from google.cloud import storage

st.set_page_config(page_title="Riyadh Weather Analytics", page_icon="🌤️",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.stApp {background:#F5F7FC;color:#172554;}
.block-container {max-width:1450px;padding-top:2rem;padding-bottom:3rem;}
h1,h2,h3 {color:#14213D!important;font-family:Arial,sans-serif;}
h1 {font-size:2.4rem!important;font-weight:800!important;}
h2 {font-size:1.5rem!important;font-weight:700!important;}
div[data-testid="stMetric"] {
background:white;padding:20px;border-radius:16px;
border:1px solid #E2E8F0;box-shadow:0 4px 16px rgba(0,0,0,0.035);}
div[data-testid="stMetricLabel"] {color:#64748B;font-size:14px;}
div[data-testid="stMetricValue"] {color:#0F172A;font-size:28px;font-weight:800;}
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:18px;}
.section-title {font-size:23px;font-weight:750;color:#14213D;
margin-top:20px;margin-bottom:10px;}
.subtitle {color:#64748B;font-size:15px;margin-bottom:22px;}
.footer {color:#94A3B8;text-align:center;font-size:12px;padding-top:20px;}
</style>
""", unsafe_allow_html=True)

BUCKET_NAME = "riyadh-weather-ml-data"
EXPERIMENT_PATH = "predictions/experiment-8016-features"

@st.cache_data(ttl=300)
def load_data(bucket_name, path):
    bucket = storage.Client().bucket(bucket_name)
    def read(name):
        return pd.read_csv(BytesIO(
            bucket.blob(f"{path}/{name}").download_as_bytes()))
    pred, actual, metrics = (
        read(f"{name}_8016.csv") for name in ("chronos", "actual", "metrics"))
    for df in (pred, actual):
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return pred, actual, metrics

try:
    predictions, actual, metrics = load_data(BUCKET_NAME, EXPERIMENT_PATH)
except Exception as error:
    st.error("Unable to load data from Google Cloud Storage.")
    st.exception(error)
    st.stop()

def prepare(target):
    pred = predictions.loc[predictions["target_name"] == target,
                           ["item_id", "time", "predictions"]]
    real = actual[["item_id", "time", target]]
    return pd.merge(
        pred.rename(columns={"predictions": "predicted"}),
        real.rename(columns={target: "actual"}),
        on=["item_id", "time"], how="inner", validate="one_to_one"
    ).sort_values("time").reset_index(drop=True)

def get_metrics(target):
    rows = metrics.loc[metrics["target"] == target]
    return rows.iloc[0].copy() if not rows.empty else None

def value(row, key, unit="", integer=False):
    if row is None or pd.isna(row[key]):
        return "N/A"
    number = row[key]
    return str(int(number)) if integer else f"{number:.{6 if unit == ' mm' else 3}f}{unit}"

def title(text):
    st.markdown(f'<div class="section-title">{text}</div>',
                unsafe_allow_html=True)

def chart(data, heading, unit, color):
    fig = go.Figure()
    for column, name, shade, width, dash in [
        ("actual", "Actual", color, 3, "solid"),
        ("predicted", "Predicted", "#F59E0B", 2.5, "dash")
    ]:
        fig.add_trace(go.Scatter(
            x=data["time"], y=data[column], mode="lines", name=name,
            line=dict(color=shade, width=width, dash=dash)))
    fig.update_layout(
        title=dict(text=heading, font=dict(size=17, color="#14213D")),
        height=420, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color="#475569"),
        xaxis=dict(title="Time (UTC)", showgrid=True, gridcolor="#F1F5F9"),
        yaxis=dict(title=unit, showgrid=True, gridcolor="#E2E8F0", zeroline=False),
        legend=dict(orientation="h", y=1.12, x=0),
        hovermode="x unified", margin=dict(l=35, r=25, t=75, b=40))
    return fig

temperature, rain = prepare("temperature_2m"), prepare("rain")
temp_metrics, rain_metrics = get_metrics("temperature_2m"), get_metrics("rain")

valid = rain[["actual", "predicted"]].apply(
    pd.to_numeric, errors="coerce").dropna()
if rain_metrics is not None and not valid.empty:
    errors = valid["actual"] - valid["predicted"]
    rain_metrics["MAE"] = errors.abs().mean()
    rain_metrics["RMSE"] = errors.pow(2).mean() ** 0.5
    metrics.loc[metrics["target"] == "rain", ["MAE", "RMSE"]] = [
        rain_metrics["MAE"], rain_metrics["RMSE"]]

st.markdown("# 🌤️ Riyadh Weather Analytics")
st.markdown(
    '<div class="subtitle">Machine Learning Weather Forecasting • '
    'Amazon Chronos-2 • Experiment 8016 with Features</div>',
    unsafe_allow_html=True)
st.info("Experiment 8016 with Feature Engineering | 168-Hour Weather Forecast | Riyadh")

title("📊 Forecast Overview")
overview = [
    ("📍 Location", "Riyadh"),
    ("🕒 Test Period", "168 Hours"),
    ("🌡️ Temperature MAE", value(temp_metrics, "MAE", " °C")),
    ("🎯 Temperature RMSE", value(temp_metrics, "RMSE", " °C"))]
for col, (label, result) in zip(st.columns(4), overview):
    with col:
        st.metric(label, result)

def forecast(heading, data, chart_title, unit, color, row, rainfall=False):
    title(heading)
    with st.container(border=True):
        st.plotly_chart(chart(data, chart_title, unit, color), width="stretch")
        suffix = " mm" if rainfall else " °C"
        cards = [("MAE", value(row, "MAE", suffix)),
                 ("RMSE", value(row, "RMSE", suffix))]
        if rainfall:
            rainy = value(row, "nonzero_actual_hours", integer=True)
            hours = value(row, "hours", integer=True)
            cards.append(("Rainy Hours", f"{rainy} / {hours}"))
        else:
            cards.append(("Test Hours", value(row, "hours", integer=True)))
        for col, (label, result) in zip(st.columns(3), cards):
            with col:
                st.metric(label, result)
        if not rainfall:
            st.caption("MAE and RMSE are taken from the "
                       "feature-engineered model evaluation results.")

forecast("🌡️ Temperature Forecast", temperature,
         "Actual vs Predicted Temperature", "Temperature (°C)",
         "#2563EB", temp_metrics)
forecast("🌧️ Rainfall Forecast", rain,
         "Actual vs Predicted Rainfall", "Rainfall (mm)",
         "#0891B2", rain_metrics, rainfall=True)

title("📋 Evaluation Summary")
summary = metrics.copy()
summary["Target"] = summary["target"].replace(
    {"temperature_2m": "Temperature", "rain": "Rainfall"})
summary = summary.rename(columns={
    "hours": "Test Hours", "nonzero_actual_hours": "Nonzero Actual Hours"})
st.dataframe(
    summary[["Target", "Test Hours", "MAE", "RMSE", "Nonzero Actual Hours"]],
    hide_index=True, width="stretch",
    column_config={
        "MAE": st.column_config.NumberColumn("MAE", format="%.6f"),
        "RMSE": st.column_config.NumberColumn("RMSE", format="%.6f")})

st.divider()
st.markdown(
    '<div class="footer">Open-Meteo Weather Data | '
    'Google Cloud Storage | Amazon Chronos-2 | '
    'Experiment 8016 with Features</div>',
    unsafe_allow_html=True)
