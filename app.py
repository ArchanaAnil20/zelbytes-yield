import streamlit as st
import numpy as np
import pandas as pd

from src.predict import predict_yield
from src.logging_utils import log_prediction

# --------------------------------
# Page configuration
# --------------------------------
st.set_page_config(
    page_title="Mushroom Yield Forecast",
    page_icon="🌱",
    layout="centered"
)

# --------------------------------
# Title
# --------------------------------
st.title("🌱 Polyhouse Yield Predictor")
st.caption("Agritech environmental forecasting from sensor data")

# --------------------------------
# Sidebar inputs
# --------------------------------
with st.sidebar:
    st.header("Sensor Readings")

    temp = st.slider(
        "Temperature (°C)",
        min_value=10.0,
        max_value=35.0,
        value=22.0,
        step=0.1
    )

    humid = st.slider(
        "Humidity (%)",
        min_value=50.0,
        max_value=100.0,
        value=88.0,
        step=0.5
    )

    co2 = st.slider(
        "CO₂ (ppm)",
        min_value=400,
        max_value=2000,
        value=900,
        step=10
    )

# --------------------------------
# Warnings
# --------------------------------
if temp < 15 or temp > 30:
    st.warning("Temperature is outside the normal training range.")

if humid < 70 or humid > 95:
    st.warning("Humidity is outside the normal training range.")

if co2 < 600 or co2 > 1200:
    st.warning("CO₂ concentration is outside the normal training range.")

# --------------------------------
# Prediction
# --------------------------------
if st.button("Predict Yield"):
    try:
        with st.spinner("Predicting yield..."):
            kg = predict_yield(temp, humid, co2)

            # Log prediction
            log_prediction(temp, humid, co2, kg)

        st.metric(
            label="Estimated Daily Yield",
            value=f"{kg:.2f} kg"
        )

    except Exception:
        st.error(
            "Model files are missing. Please check project artifacts."
        )

# --------------------------------
# Sensitivity chart
# --------------------------------
st.subheader("What-if: Humidity Sweep")

temp_fixed = 22.0
co2_fixed = 900

humid_range = np.linspace(70, 98, 29)

preds = [
    predict_yield(temp_fixed, h, co2_fixed)
    for h in humid_range
]

chart_df = pd.DataFrame({
    "Humidity (%)": humid_range,
    "Predicted Yield (kg)": preds
})

st.line_chart(
    chart_df,
    x="Humidity (%)",
    y="Predicted Yield (kg)"
)

# --------------------------------
# Model information
# --------------------------------
with st.expander("Model Information"):
    st.markdown("""
- **Model:** Tuned Random Forest
- **Test MAE:** 1.2 kg/day
- **Training data:** Polyhouse sensors Jan–Dec 2024
""")

# --------------------------------
# Recent prediction logs
# --------------------------------
with st.expander("Recent Prediction Logs"):
    try:
        logs_df = pd.read_csv("logs/predictions.csv")
        st.dataframe(logs_df.tail(10))
    except:
        st.info("No prediction logs available yet.")