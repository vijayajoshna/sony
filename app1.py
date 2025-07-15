Python 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
>>> import streamlit as st
... import numpy as np
... import joblib
... 
... # Load saved model and scaler
... model = joblib.load('diabetes_model.pkl')
... scaler = joblib.load('scaler.pkl')
... 
... def calculate_bmi(weight, height_cm):
...     height_m = height_cm / 100
...     return round(weight / (height_m ** 2), 2)
... 
... def advanced_prediction():
...     st.subheader("🧪 Advanced Health Data")
... 
...     weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0)
...     height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0)
...     age = st.number_input("Age", min_value=10, max_value=100)
...     glucose = st.number_input("Glucose Level (mg/dL)", min_value=50.0, max_value=300.0)
...     bp = st.number_input("Blood Pressure (mm Hg)", min_value=50.0, max_value=200.0)
...     pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20)
...     insulin = st.number_input("Insulin Level", min_value=0.0, max_value=900.0)
...     skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0)
... 
...     if st.button("Predict Diabetes (Advanced)"):
...         bmi = calculate_bmi(weight, height)
...         X_input = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, age]])
...         X_scaled = scaler.transform(X_input)
...         prob = model.predict_proba(X_scaled)[0][1] * 100
...         prediction = "Likely Diabetic" if prob >= 60 else ("Borderline Risk" if prob >= 30 else "Non-Diabetic")
...         st.success(f"Prediction: {prediction} ({prob:.2f}%)")
...         st.info(f"Calculated BMI: {bmi}")
... 
... def symptom_based_prediction():
...     st.subheader("🔍 Quick Symptom Checker")
...     st.markdown("(Answer based on how you feel)")
... 
...     q1 = st.checkbox("Frequent urination")
...     q2 = st.checkbox("Excessive thirst")
...     q3 = st.checkbox("Unexplained weight loss")
    q4 = st.checkbox("Blurred vision")
    q5 = st.checkbox("Fatigue or weakness")
    q6 = st.checkbox("Slow healing wounds")
    q7 = st.checkbox("Tingling in hands/feet")

    symptoms = [q1, q2, q3, q4, q5, q6, q7]
    score = sum(symptoms)

    if st.button("Check Symptoms"):
        if score >= 5:
            st.error("⚠️ High Risk of Diabetes (Based on Symptoms)")
        elif score >= 3:
            st.warning("⚠️ Moderate Risk — Consult a Doctor")
        else:
            st.success("✅ Low Risk Based on Symptoms")

# ---------- Streamlit UI Layout ----------
st.title("🩺 Diabetes Prediction App")

mode = st.radio("Choose Mode", ["🧪 Enter Medical Data", "🔍 Check with Symptoms"])

if mode == "🧪 Enter Medical Data":
    advanced_prediction()
else:
    symptom_based_prediction()
