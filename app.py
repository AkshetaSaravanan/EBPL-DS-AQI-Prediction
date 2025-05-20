import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os

# --- Function to predict AQI for any date (past or future) ---
def predict_aqi_for_date(city, date):
    model_path = f'models/{city}_model.pkl'
    model = joblib.load(model_path)
    date_ordinal = pd.DataFrame({'Date_ordinal': [date.toordinal()]})
    predicted_aqi = model.predict(date_ordinal)[0]
    return predicted_aqi

# --- Page config ---
st.set_page_config(page_title="AQI Predictor India", layout="centered", page_icon="🌿")

# --- Title Section ---
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #4CAF50;'>🌍 EBPL-DS: Air Quality Predictor</h1>
        <p style='font-size:18px;'>Predicting AQI for Indian cities using AI + Streamlit (2020–2024 and beyond)</p>
    </div>
    <hr>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("🧭 Navigation")
st.sidebar.info("Select a city and date to get AQI prediction.")
st.sidebar.title("📌 Did You Know?")
st.sidebar.warning("Delhi's AQI crossed 500+ in winters of 2022, classifying it as 'Hazardous'.")

# --- Main Inputs ---
cities = ['Bangalore', 'Delhi', 'Mumbai', 'Chennai', 'Kolkata']
col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("🏙️ Select City", cities)

with col2:
    # Allow dates from 2020 up to 2030 (future dates enabled)
    date_input = st.date_input("📅 Choose Date", min_value=datetime(2020, 1, 1), max_value=datetime(2030, 12, 31))

# --- AQI Category Logic ---
def get_aqi_feedback(aqi):
    if aqi <= 50:
        return '🟢 Good – Air quality is satisfactory'
    elif aqi <= 100:
        return '🟡 Moderate – Acceptable, but may affect sensitive individuals'
    elif aqi <= 150:
        return '🟠 Poor – May cause breathing discomfort'
    elif aqi <= 200:
        return '🔴 Unhealthy – Everyone may feel ill effects'
    elif aqi <= 300:
        return '🟣 Very Unhealthy – Medical alert level'
    else:
        return '⚫ Hazardous – Emergency conditions'

# --- Predict Button ---
st.markdown("### 🔍 AQI Prediction Result")
if st.button("🚀 Predict AQI Now"):
    try:
        prediction = predict_aqi_for_date(city, date_input)
        feedback = get_aqi_feedback(prediction)

        st.success(f"**Predicted AQI for {city} on {date_input}: {prediction:.2f}**")
        st.markdown(f"### {feedback}")

        # --- Show historical trend graph for 2020-2024 ---
        graph_path = f'images/{city}_aqi_plot.png'
        if os.path.exists(graph_path):
            st.image(graph_path, caption=f"{city} AQI Trend (2020–2024)", use_container_width=True)
        else:
            st.info("AQI trend graph not found for this city.")

    except FileNotFoundError:
        st.error("Model file not found. Please make sure models are trained.")

# --- Footer ---
st.markdown("""
<hr>
<p style='text-align: center; font-size: 14px;'>
    📘 <a href='https://github.com/AkshetaSaravanan/EBPL-DS-AQI-Prediction/tree/main' target='_blank'>View Source on GitHub</a> |
    Built with ❤️ by TEAM SKODA SUPERB | EBPL-DS Project |
</p>
""", unsafe_allow_html=True)
