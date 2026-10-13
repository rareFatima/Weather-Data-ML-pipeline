from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from google.cloud import storage


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Riyadh Weather Analytics",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================
# 2. CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>
    .stApp {
        background: #F5F7FC;
        color: #172554;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #14213D !important;
        font-family: 'Arial', sans-serif;
    }

    h1 {
        font-size: 2.4rem !important;
        font-weight: 800 !important;
    }

    h2 {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetric"] {
        background: white;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.035);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748B;
        font-size: 14px;
    }

    div[data-testid="stMetricValue"] {
        color: #0F172A;
        font-size: 28px;
        font-weight: 800;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 750;
        color: #14213D;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .subtitle {
        color: #64748B;
        font-size: 15px;
        margin-bottom: 22px;
    }

    .footer {
        color: #94A3B8;
        text-align: center;
        font-size: 12px;
        padding-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. GOOGLE CLOUD CONFIGURATION
# ==========================================

BUCKET_NAME = "riyadh-weather-ml-data"

# Change this path when switching experiments
EXPERIMENT_PATH = "predictions/experiment-8016-features"

PREDICTIONS_FILE = (
    f"{EXPERIMENT_PATH}/chronos_8016.csv"
)

ACTUAL_FILE = (
    f"{EXPERIMENT_PATH}/actual_8016.csv"
)

METRICS_FILE = (
    f"{EXPERIMENT_PATH}/metrics_8016.csv"
)


# ==========================================
# 4. LOAD DATA FROM GOOGLE CLOUD
# ==========================================

@st.cache_data(ttl=300)
def load_data(
    bucket_name,
    pred_path,
    actual_path,
    metrics_path
):

    client = storage.Client()

    bucket = client.bucket(bucket_name)

    def read_csv(blob_path):

        blob = bucket.blob(blob_path)

        content = blob.download_as_bytes()

        return pd.read_csv(BytesIO(content))

    predictions = read_csv(pred_path)

    actual = read_csv(actual_path)

    metrics = read_csv(metrics_path)

    predictions["time"] = pd.to_datetime(
        predictions["time"],
        utc=True
    )

    actual["time"] = pd.to_datetime(
        actual["time"],
        utc=True
    )

    return predictions, actual, metrics


try:

    predictions, actual, metrics = load_data(
        BUCKET_NAME,
        PREDICTIONS_FILE,
        ACTUAL_FILE,
        METRICS_FILE
    )

except Exception as error:

    st.error(
        "Unable to load data from Google Cloud Storage."
    )

    st.exception(error)

    st.stop()


# ==========================================
# 5. PREPARE TARGET DATA
# ==========================================

def prepare_target_data(target):

    target_predictions = predictions[
        predictions["target_name"] == target
    ][
        [
            "item_id",
            "time",
            "predictions"
        ]
    ].copy()

    target_predictions = target_predictions.rename(
        columns={
            "predictions": "predicted"
        }
    )

    target_actual = actual[
        [
            "item_id",
            "time",
            target
        ]
    ].copy()

    target_actual = target_actual.rename(
        columns={
            target: "actual"
        }
    )

    merged = pd.merge(
        target_predictions,
        target_actual,
        on=[
            "item_id",
            "time"
        ],
        how="inner",
        validate="one_to_one"
    )

    merged = merged.sort_values(
        "time"
    ).reset_index(drop=True)

    return merged


temperature = prepare_target_data(
    "temperature_2m"
)

rain = prepare_target_data(
    "rain"
)


# ==========================================
# 6. GET EVALUATION METRICS
# ==========================================

def get_metrics(target):

    target_metrics = metrics[
        metrics["target"] == target
    ]

    if target_metrics.empty:
        return None

    return target_metrics.iloc[0]


temp_metrics = get_metrics(
    "temperature_2m"
)

rain_metrics = get_metrics(
    "rain"
)


# ==========================================
# 7. HEADER
# ==========================================

st.markdown(
    "# 🌤️ Riyadh Weather Analytics"
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Weather Forecasting • '
    'Amazon Chronos-2 • '
    'Experiment 8016 with Features'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Experiment 8016 with Feature Engineering | "
    "168-Hour Weather Forecast | Riyadh"
)


# ==========================================
# 8. FORECAST OVERVIEW
# ==========================================

st.markdown(
    '<div class="section-title">'
    '📊 Forecast Overview'
    '</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "📍 Location",
        "Riyadh"
    )


with col2:

    st.metric(
        "🕒 Test Period",
        "168 Hours"
    )


with col3:

    st.metric(
        "🌡️ Temperature MAE",
        (
            f"{temp_metrics['MAE']:.3f} °C"
            if temp_metrics is not None
            else "N/A"
        )
    )


with col4:

    st.metric(
        "🎯 Temperature RMSE",
        (
            f"{temp_metrics['RMSE']:.3f} °C"
            if temp_metrics is not None
            else "N/A"
        )
    )


# ==========================================
# 9. CHART FUNCTION
# ==========================================

def create_chart(
    data,
    title,
    unit,
    color
):

    fig = go.Figure()

    # Actual values
    fig.add_trace(
        go.Scatter(
            x=data["time"],
            y=data["actual"],
            mode="lines",
            name="Actual",
            line=dict(
                color=color,
                width=3
            )
        )
    )

    # Predicted values
    fig.add_trace(
        go.Scatter(
            x=data["time"],
            y=data["predicted"],
            mode="lines",
            name="Predicted",
            line=dict(
                color="#F59E0B",
                width=2.5,
                dash="dash"
            )
        )
    )

    fig.update_layout(

        title=dict(
            text=title,
            font=dict(
                size=17,
                color="#14213D"
            )
        ),

        height=420,

        paper_bgcolor="white",

        plot_bgcolor="white",

        font=dict(
            color="#475569"
        ),

        xaxis=dict(
            title="Time (UTC)",
            showgrid=True,
            gridcolor="#F1F5F9"
        ),

        yaxis=dict(
            title=unit,
            showgrid=True,
            gridcolor="#E2E8F0",
            zeroline=False
        ),

        legend=dict(
            orientation="h",
            y=1.12,
            x=0
        ),

        hovermode="x unified",

        margin=dict(
            l=35,
            r=25,
            t=75,
            b=40
        )
    )

    return fig


# ==========================================
# 10. TEMPERATURE SECTION
# ==========================================

st.markdown(
    '<div class="section-title">'
    '🌡️ Temperature Forecast'
    '</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    st.plotly_chart(
        create_chart(
            temperature,
            "Actual vs Predicted Temperature",
            "Temperature (°C)",
            "#2563EB"
        ),
        width="stretch"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "MAE",
            (
                f"{temp_metrics['MAE']:.3f} °C"
                if temp_metrics is not None
                else "N/A"
            )
        )

    with c2:

        st.metric(
            "RMSE",
            (
                f"{temp_metrics['RMSE']:.3f} °C"
                if temp_metrics is not None
                else "N/A"
            )
        )

    with c3:

        st.metric(
            "Test Hours",
            (
                str(int(temp_metrics["hours"]))
                if temp_metrics is not None
                else "N/A"
            )
        )

    st.caption(
        "MAE and RMSE are taken from the "
        "feature-engineered model evaluation results."
    )


# ==========================================
# 11. RAINFALL SECTION
# ==========================================

st.markdown(
    '<div class="section-title">'
    '🌧️ Rainfall Forecast'
    '</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    st.plotly_chart(
        create_chart(
            rain,
            "Actual vs Predicted Rainfall",
            "Rainfall (mm)",
            "#0891B2"
        ),
        width="stretch"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "MAE",
            (
                f"{rain_metrics['MAE']:.3f} mm"
                if rain_metrics is not None
                else "N/A"
            )
        )

    with c2:

        st.metric(
            "RMSE",
            (
                f"{rain_metrics['RMSE']:.3f} mm"
                if rain_metrics is not None
                else "N/A"
            )
        )

    with c3:

        rainy_hours = (
            int(rain_metrics["nonzero_actual_hours"])
            if rain_metrics is not None
            else None
        )

        total_hours = (
            int(rain_metrics["hours"])
            if rain_metrics is not None
            else None
        )

        st.metric(
            "Rainy Hours",
            (
                f"{rainy_hours} / {total_hours}"
                if rainy_hours is not None
                else "N/A"
            )
        )


# ==========================================
# 12. EVALUATION SUMMARY
# ==========================================

st.markdown(
    '<div class="section-title">'
    '📋 Evaluation Summary'
    '</div>',
    unsafe_allow_html=True
)

summary = metrics.copy()

summary["Target"] = summary[
    "target"
].replace({
    "temperature_2m": "Temperature",
    "rain": "Rainfall"
})

summary = summary.rename(
    columns={
        "hours": "Test Hours",
        "nonzero_actual_hours":
            "Nonzero Actual Hours"
    }
)

st.dataframe(
    summary[
        [
            "Target",
            "Test Hours",
            "MAE",
            "RMSE",
            "Nonzero Actual Hours"
        ]
    ],
    hide_index=True,
    width="stretch"
)


# ==========================================
# 13. FOOTER
# ==========================================

st.divider()

st.markdown(
    '<div class="footer">'
    'Open-Meteo Weather Data | '
    'Google Cloud Storage | '
    'Amazon Chronos-2 | '
    'Experiment 8016 with Features'
    '</div>',
    unsafe_allow_html=True
)
