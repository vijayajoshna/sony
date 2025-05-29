import streamlit as st
import numpy as np
import joblib

# Load model and scaler
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

st.title("🧪 Diabetes Prediction System")
st.markdown("Enter the details below to check your diabetes risk:")

# Input fields
pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose Level (mg/dL)", min_value=0.0)
bp = st.number_input("Blood Pressure (mm Hg)", min_value=0.0)
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0)
insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0.0)
height_cm = st.number_input("Height (cm)", min_value=0.0)
weight = st.number_input("Weight (kg)", min_value=0.0)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, step=0.01)
age = st.number_input("Age (years)", min_value=1, step=1)

if st.button("Predict"):
    if height_cm == 0:
        st.error("Height can't be 0")
    else:
        height_m = height_cm / 100
        bmi = weight / (height_m ** 2)
        st.write(f"Calculated BMI: **{bmi:.2f}**")

        input_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
        try:
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            result = "🟢 Likely Non-Diabetic" if prediction[0] == 0 else "🔴 Likely Diabetic"
            st.success(f"Prediction: {result}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
