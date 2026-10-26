import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page settings
st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="centered"
)

# Load trained model
MODEL_PATH = Path(__file__).with_name("student_performance_model.pkl")
model = joblib.load(MODEL_PATH)

# Custom design
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f3e8ff, #e0e7ff);
}
.main-title {
    text-align: center;
    color: #5b21b6;
    font-size: 36px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #4b5563;
    font-size: 17px;
}
.result-card {
    background-color: white;
    padding: 22px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# Heading
st.markdown(
    '<div class="main-title">🎓 Student Performance AI</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Machine Learning Based Student Result Prediction</div>',
    unsafe_allow_html=True
)

st.divider()

st.subheader("👩‍🎓 Enter Student Details")
st.write("Fill in the details below to predict the student's result.")

# Input fields
study_hours = st.number_input(
    "📚 Study Hours Per Day",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

attendance = st.slider(
    "📅 Attendance Percentage",
    min_value=0,
    max_value=100,
    value=75
)

previous_marks = st.slider(
    "📝 Previous Marks",
    min_value=0,
    max_value=100,
    value=60
)

# Prediction
if st.button("🔍 Predict Student Result", use_container_width=True):
    student = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Previous_Marks": [previous_marks]
    })

    prediction = model.predict(student)[0]

    st.divider()
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.success("🎉 Predicted Result: PASS")
        st.write("The model predicts that the student will pass.")
    else:
        st.error("📚 Predicted Result: FAIL")
        st.write("The model predicts that the student may need additional support.")

    st.info(
        f"Study Hours: {study_hours} | "
        f"Attendance: {attendance}% | "
        f"Previous Marks: {previous_marks}"
    )

st.divider()

st.caption(
    "⚠️ Educational demonstration only. "
    "The model uses a small sample dataset, so its predictions are not "
    "reliable guarantees of actual academic performance."
)

st.markdown(
    "<center>Built with Python, Streamlit and Scikit-learn</center>",
    unsafe_allow_html=True
)