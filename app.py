import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ---------- Page configuration ----------
st.set_page_config(
    page_title="Hospital Waiting Time Predictor",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS styling ----------
st.markdown("""
    <style>
    .main { background-color: #f4f9fb; }
    .title-text {
        font-size: 38px; font-weight: 800; color: #0b5394;
        text-align: center; margin-bottom: 0px;
    }
    .subtitle-text {
        font-size: 16px; color: #4a4a4a; text-align: center; margin-bottom: 25px;
    }
    .stButton>button {
        background-color: #0b5394; color: white; font-weight: 600;
        border-radius: 10px; padding: 10px 24px; border: none; width: 100%;
    }
    .stButton>button:hover { background-color: #073763; color: white; }
    .result-box {
        background-color: #d9ead3; border-left: 6px solid #38761d;
        padding: 20px; border-radius: 10px; font-size: 22px; font-weight: 700;
        color: #274e13; text-align: center; margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Train the model fresh, once, and cache it ----------
@st.cache_resource
def load_and_train():
    df = pd.read_excel("hospital_waiting_time_dataset.xlsx")
    df["arrival_hour"] = df["arrival_time"].str.split(":").str[0].astype(int)

    encoders = {}
    for col in ["department", "day_of_week", "emergency_status"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le

    feature_cols = [
        "department_enc", "arrival_hour", "number_of_patients",
        "doctor_availability", "day_of_week_enc", "emergency_status_enc",
        "previous_waiting_time",
    ]
    X = df[feature_cols]
    y = df["waiting_time_minutes"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    return model, encoders

model, encoders = load_and_train()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("ℹ️ About this App")
    st.write(
        "This tool uses a Machine Learning model (Random Forest Regressor) "
        "to estimate how long a patient is likely to wait before receiving service."
    )
    st.write("Fill in the patient details on the right and click **Predict**.")
    st.markdown("---")
    st.caption("EASTC — BDTS Machine Learning Project")

# ---------- Main title ----------
st.markdown('<p class="title-text">🏥 Hospital Waiting Time Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Enter patient details below to estimate the expected waiting time</p>', unsafe_allow_html=True)

# ---------- Input form ----------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        department = st.selectbox("🏬 Department", encoders["department"].classes_)
        arrival_hour = st.slider("🕒 Arrival Hour (24h)", 0, 23, 9)
        number_of_patients = st.number_input("👥 Number of Patients in Queue", min_value=0, max_value=100, value=15)
        doctor_availability = st.number_input("🩺 Doctors Available", min_value=1, max_value=20, value=5)

    with col2:
        day_of_week = st.selectbox("📅 Day of Week", encoders["day_of_week"].classes_)
        emergency_status = st.selectbox("🚨 Emergency Status", encoders["emergency_status"].classes_)
        previous_waiting_time = st.number_input("⏱️ Previous Waiting Time (minutes)", min_value=0.0, value=40.0)

    submitted = st.form_submit_button("🔍 Predict Waiting Time")

# ---------- Prediction ----------
if submitted:
    input_df = pd.DataFrame([{
        "department_enc": encoders["department"].transform([department])[0],
        "arrival_hour": arrival_hour,
        "number_of_patients": number_of_patients,
        "doctor_availability": doctor_availability,
        "day_of_week_enc": encoders["day_of_week"].transform([day_of_week])[0],
        "emergency_status_enc": encoders["emergency_status"].transform([emergency_status])[0],
        "previous_waiting_time": previous_waiting_time,
    }])

    prediction = model.predict(input_df)[0]

    st.markdown(
        f'<div class="result-box">Estimated Waiting Time: {prediction:.0f} minutes</div>',
        unsafe_allow_html=True
    )

    if emergency_status == "Emergency":
        st.info("⚡ Emergency cases are typically prioritized and seen faster.")
    elif number_of_patients > 30:
        st.warning("⚠️ High patient volume detected — waiting time may be longer than usual.")
