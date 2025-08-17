import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("school_performance_gb_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("School Performance Band Prediction")
st.write("Enter school metrics to predict the performance band (Low/Medium/High)")

# Input fields
attendance_rate = st.number_input("Attendance Rate (%)", 0.0, 100.0, 90.0)
staff_performance = st.number_input("Staff Performance (1-5)", 0.0, 5.0, 4.0)
parent_feedback = st.number_input("Parent Feedback (1-5)", 0.0, 5.0, 3.5)
budget_per_student = st.number_input("Budget per Student (LKR)", 0.0, 10000.0, 400.0)
extracurricular_count = st.number_input("Extracurricular Activities Count", 0, 10, 2)
class_size = st.number_input("Class Size", 1, 100, 30)
internal_assessment = st.number_input("Internal Assessment Score (%)", 0.0, 100.0, 75.0)
exam_score = st.number_input("Exam Score (%)", 0.0, 100.0, 70.0)
term = st.selectbox("Term", ["Term 1", "Term 2", "Term 3"])

# Manual encoding for term
term_mapping = {"Term 1": 0, "Term 2": 1, "Term 3": 2}
term_encoded = term_mapping[term]

# Manual decoding for performance_band
performance_mapping = {0: "High", 1: "Low", 2: "Medium"}

# Prediction button
if st.button("Predict Performance Band"):
    input_data = np.array([[attendance_rate, staff_performance, parent_feedback,
                            budget_per_student, extracurricular_count, class_size,
                            internal_assessment, exam_score, term_encoded]])
    input_scaled = scaler.transform(input_data)
    prediction_encoded = model.predict(input_scaled)[0]
    predicted_band = performance_mapping[prediction_encoded]

    st.success(f"Predicted Performance Band: {predicted_band}")
