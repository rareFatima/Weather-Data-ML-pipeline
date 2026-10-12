from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Riyadh Weather Analytics",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================
# CUSTOM DESIGN
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
# DATA CONFIGURATION
# ==========================================

DATA_DIR = Path("data/dashboard")

PREDICTIONS_FILE = DATA_DIR / "chronos_8016.csv"
ACTUAL_FILE = DATA_DIR / "actual_8016.csv"
METRICS_FILE = DATA_DIR / "metrics_8016.csv"


@st.cache_data
def load_data(pred_path, actual_path, metrics_path):

    predictions = pd.read_csv(pred_path)
    actual = pd.read_csv(actual_path)
    metrics = pd.read_csv(metrics_path)

    predictions["time"] = pd.to_datetime(
        predictions["time"], utc=True
    )

    actual["time"] = pd.to_datetime(
        actual["time"], utc=True
    )

    return predictions, actual, metrics


try:
    predictions, actual, metrics = load_data(
        str(PREDICTIONS_FILE),
        str(ACTUAL_FILE),
        str(METRICS_FILE)
    )

except Exception as error:
    st.error(f"Unable to load dashboard data: {error}")
    st.stop()


# ==========================================
# PREPARE DATA
# ==========================================

def prepare_target_data(target):

    pred = predictions[
        predictions["target_name"] == target
    ][["item_id", "time", "predictions"]].copy()

    pred = pred.rename(
        columns={"predictions": "predicted"}
    )

    real = actual[
        ["item_id", "time", target]
    ].copy()

    real = real.rename(
        columns={target: "actual"}
    )

    merged = pd.merge(
        pred,
        real,
        on=["item_id", "time"],
        how="inner",
        validate="one_to_one"
    )

    merged = merged.sort_values("time")

    return merged


temperature = prepare_target_data("temperature_2m")
rain = prepare_target_data("rain")


# ==========================================
# CALCULATE PERCENTAGE METRICS
# ==========================================

def calculate_mape(data):

    valid = data[
        (data["actual"].notna())
        & (data["predicted"].notna())
        & (data["actual"] != 0)
    ]

    if valid.empty:
        return None

    percentage_error = (
        np.abs(
            (valid["actual"] - valid["predicted"])
            / valid["actual"]
        ) * 100
    )

    return float(percentage_error.mean())


temperature_mape = calculate_mape(temperature)

temperature_performance = (
    max(0.0, 100.0 - temperature_mape)
    if temperature_mape is not None
    else None
)


# ==========================================
# METRICS HELPERS
# ==========================================

def get_metrics(target):

    row = metrics[
        metrics["target"] == target
    ]

    if row.empty:
        return None

    return row.iloc[0]


temp_metrics = get_metrics("temperature_2m")
rain_metrics = get_metrics("rain")


# ==========================================
# HEADER
# ==========================================

st.markdown(
    "# 🌤️ Riyadh Weather Analytics"
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Weather Forecasting • '
    'Amazon Chronos-2 • Experiment 8016'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Experiment 8016 | 168-hour forecast | "
    "Results before the new feature engineering inputs."
)


# ==========================================
# SUMMARY CARDS
# ==========================================

st.markdown(
    '<div class="section-title">📊 Forecast Overview</div>',
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
        f"{temp_metrics['MAE']:.3f} °C"
        if temp_metrics is not None else "N/A"
    )

with col4:
    st.metric(
        "🎯 Temperature Performance",
        f"{temperature_performance:.2f}%"
        if temperature_performance is not None
        else "N/A",
        help=(
            "Calculated as 100 - MAPE. "
            "This is an illustrative performance indicator, "
            "not classification accuracy."
        )
    )


# ==========================================
# CHART FUNCTION
# ==========================================

def create_chart(data, title, unit, color):

    fig = go.Figure()

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
# TEMPERATURE SECTION
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
        use_container_width=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "MAE",
            f"{temp_metrics['MAE']:.3f} °C"
            if temp_metrics is not None else "N/A"
        )

    with c2:
        st.metric(
            "RMSE",
            f"{temp_metrics['RMSE']:.3f} °C"
            if temp_metrics is not None else "N/A"
        )

    with c3:
        st.metric(
            "MAPE",
            f"{temperature_mape:.2f}%"
            if temperature_mape is not None else "N/A"
        )

    st.caption(
        "MAE and RMSE are taken from the evaluation results. "
        "MAPE is calculated from the matched actual "
        "and predicted temperature values."
    )


# ==========================================
# RAINFALL SECTION
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
        use_container_width=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "MAE",
            f"{rain_metrics['MAE']:.3f} mm"
            if rain_metrics is not None else "N/A"
        )

    with c2:
        st.metric(
            "RMSE",
            f"{rain_metrics['RMSE']:.3f} mm"
            if rain_metrics is not None else "N/A"
        )

    with c3:
        st.metric(
            "Rainy Hours",
            "0 / 168"
        )

    st.warning(
        "No rainfall occurred during the test period. "
        "Rainfall prediction accuracy cannot be "
        "reliably evaluated using this period alone."
    )


# ==========================================
# EVALUATION SUMMARY
# ==========================================

st.markdown(
    '<div class="section-title">'
    '📋 Evaluation Summary'
    '</div>',
    unsafe_allow_html=True
)

summary = metrics.copy()

summary["Target"] = summary["target"].replace({
    "temperature_2m": "Temperature",
    "rain": "Rainfall"
})

summary = summary.rename(
    columns={
        "hours": "Test Hours",
        "nonzero_actual_hours": "Nonzero Actual Hours"
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
    use_container_width=True
)


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
    '<div class="footer">'
    'Open-Meteo Weather Data | '
    'Google Cloud Storage | '
    'Amazon Chronos-2 | '
    'Experiment 8016'
    '</div>',
    unsafe_allow_html=True
)
