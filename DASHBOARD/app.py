import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from SRC.model_utils import load_model, load_features
from SRC.anomaly_detection import (
    calculate_prediction_errors,
    detect_anomalies
)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="EnerSense AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "MODELS",
    "enerSense_histgradient_model.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "MODELS",
    "model_features.pkl"
)

PREDICTION_PATH = os.path.join(
    BASE_DIR,
    "DATA",
    "processed",
    "energy_predictions.csv"
)

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_features():
    return joblib.load(FEATURE_PATH)


@st.cache_data
def load_predictions():
    df = pd.read_csv(PREDICTION_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


model = load_model()
features = load_features()
predictions = load_predictions()

# =========================================================
# HEADER
# =========================================================

st.title("⚡ EnerSense AI")

st.markdown(
    """
    ### Smart Building Energy Forecasting & Monitoring

    AI-powered energy forecasting, anomaly detection and
    consumption insights for smarter building management.
    """
)

st.divider()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚡ EnerSense AI")

st.sidebar.markdown("### Dashboard Controls")

st.sidebar.info(
    "Building: Hog_other_Tobias\n\n"
    "Site: Hog\n\n"
    "Building Type: Animal Shelter"
)

# Date filter
min_date = predictions["timestamp"].min().date()
max_date = predictions["timestamp"].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Handle date selection
if isinstance(date_range, tuple) and len(date_range) == 2:

    start_date, end_date = date_range

    filtered = predictions[
        (predictions["timestamp"].dt.date >= start_date)
        &
        (predictions["timestamp"].dt.date <= end_date)
    ].copy()

else:
    filtered = predictions.copy()

st.sidebar.divider()

st.sidebar.markdown("### Model")

st.sidebar.success(
    "HistGradientBoosting"
)

st.sidebar.metric(
    "Model R²",
    "92.58%"
)

st.sidebar.metric(
    "Model MAE",
    "2.0751"
)

# =========================================================
# KPI CALCULATIONS
# =========================================================

if len(filtered) > 0:

    latest = filtered.iloc[-1]

    current_energy = latest["actual"]
    predicted_energy = latest["predicted"]

    average_energy = filtered["actual"].mean()

    anomaly_threshold = 5

    anomaly_count = (
        filtered["absolute_error"] > anomaly_threshold
    ).sum()

else:

    current_energy = 0
    predicted_energy = 0
    average_energy = 0
    anomaly_count = 0

# =========================================================
# KPI CARDS
# =========================================================

st.subheader("📊 Energy Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "⚡ Latest Consumption",
        f"{current_energy:.2f}"
    )

with col2:
    st.metric(
        "🔮 Latest Prediction",
        f"{predicted_energy:.2f}"
    )

with col3:
    st.metric(
        "📈 Average Consumption",
        f"{average_energy:.2f}"
    )

with col4:
    st.metric(
        "🚨 Detected Anomalies",
        int(anomaly_count)
    )

# =========================================================
# ACTUAL VS PREDICTED
# =========================================================

st.divider()

st.subheader("📈 Actual vs Predicted Energy")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=filtered["timestamp"],
        y=filtered["actual"],
        mode="lines",
        name="Actual Consumption"
    )
)

fig.add_trace(
    go.Scatter(
        x=filtered["timestamp"],
        y=filtered["predicted"],
        mode="lines",
        name="Predicted Consumption"
    )
)

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Energy Consumption",
    hovermode="x unified",
    height=500
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# ERROR ANALYSIS
# =========================================================

st.subheader("🎯 Prediction Error")

fig_error = go.Figure()

fig_error.add_trace(
    go.Scatter(
        x=filtered["timestamp"],
        y=filtered["absolute_error"],
        mode="lines",
        name="Absolute Error"
    )
)

fig_error.add_hline(
    y=5,
    line_dash="dash",
    annotation_text="Anomaly Threshold"
)

fig_error.update_layout(
    xaxis_title="Time",
    yaxis_title="Absolute Error",
    height=400
)

st.plotly_chart(
    fig_error,
    use_container_width=True
)

# =========================================================
# ANOMALY MONITORING
# =========================================================

st.divider()

st.subheader("🚨 Anomaly Monitoring")

anomalies = filtered[
    filtered["absolute_error"] > anomaly_threshold
].copy()

if len(anomalies) > 0:

    st.warning(
        f"{len(anomalies)} potential unusual consumption "
        "points detected."
    )

    display_columns = [
        "timestamp",
        "actual",
        "predicted",
        "absolute_error"
    ]

    st.dataframe(
        anomalies[display_columns]
        .sort_values(
            "absolute_error",
            ascending=False
        )
        .head(20),
        use_container_width=True
    )

else:

    st.success(
        "✅ No significant prediction anomalies detected "
        "for the selected period."
    )

# =========================================================
# DAILY CONSUMPTION
# =========================================================

st.divider()

st.subheader("📅 Daily Energy Consumption")

daily = (
    filtered
    .set_index("timestamp")["actual"]
    .resample("D")
    .mean()
    .reset_index()
)

fig_daily = go.Figure()

fig_daily.add_trace(
    go.Scatter(
        x=daily["timestamp"],
        y=daily["actual"],
        mode="lines",
        name="Daily Average"
    )
)

fig_daily.update_layout(
    xaxis_title="Date",
    yaxis_title="Average Energy Consumption",
    height=400
)

st.plotly_chart(
    fig_daily,
    use_container_width=True
)

# =========================================================
# BUILDING INFORMATION
# =========================================================

st.divider()

st.subheader("🏢 Building Information")

info1, info2, info3, info4 = st.columns(4)

with info1:
    st.metric(
        "Building",
        "Hog_other_Tobias"
    )

with info2:
    st.metric(
        "Site",
        "Hog"
    )

with info3:
    st.metric(
        "Building Type",
        "Animal Shelter"
    )

with info4:
    st.metric(
        "Area",
        "2,220.7 m²"
    )

# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

st.subheader("🤖 AI Model")

st.write(
    """
    EnerSense AI uses a **HistGradientBoosting** model to
    forecast hourly electricity consumption.
    """
)

model_col1, model_col2, model_col3 = st.columns(3)

with model_col1:
    st.metric(
        "R² Score",
        "0.9258"
    )

with model_col2:
    st.metric(
        "MAE",
        "2.0751"
    )

with model_col3:
    st.metric(
        "RMSE",
        "3.0516"
    )

with st.expander("View Model Features"):

    st.write(
        f"Total features: **{len(features)}**"
    )

    st.write(features)

# =========================================================
# RECOMMENDATIONS
# =========================================================

st.divider()

st.subheader("💡 Energy Insights")

if average_energy > 40:

    st.warning(
        """
        **High average consumption detected.**

        Consider reviewing building operating schedules,
        HVAC usage and equipment operating hours.
        """
    )

else:

    st.success(
        """
        **Energy consumption is currently within the
        observed average range.**

        Continue monitoring consumption patterns for
        unusual changes.
        """
    )

if anomaly_count > 0:

    st.info(
        f"""
        **{anomaly_count} unusual consumption points**
        were detected in the selected period.

        These periods should be reviewed to determine whether
        they are caused by weather, occupancy, equipment
        operation or other operational factors.
        """
    )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "EnerSense AI • Smart Building Energy Intelligence"
)

st.caption(
    "Built with Python, Pandas, Scikit-learn, Plotly & Streamlit"
)
