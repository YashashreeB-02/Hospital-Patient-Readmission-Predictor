import streamlit as st
import pandas as pd
import joblib


model = joblib.load("readmission_model.pkl")


st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Hospital Patient Readmission Predictor")
st.write(
    "This system predicts the risk of hospital readmission using a Machine Learning model."
)

st.markdown("---")


st.sidebar.header("Patient Details")

age_group = st.sidebar.selectbox(
    "Age Group",
    [
        "0-10", "10-20", "20-30", "30-40", "40-50",
        "50-60", "60-70", "70-80", "80-90", "90-100"
    ]
)

time_in_hospital = st.sidebar.slider(
    "Days Spent in Hospital", 1, 14, 4
)

num_lab_procedures = st.sidebar.slider(
    "Number of Lab Procedures", 1, 100, 40
)

num_medications = st.sidebar.slider(
    "Number of Medications", 1, 50, 10
)

number_inpatient = st.sidebar.slider(
    "Previous Inpatient Visits", 0, 10, 1
)


age_mapping = {
    "0-10": 0, "10-20": 1, "20-30": 2, "30-40": 3,
    "40-50": 4, "50-60": 5, "60-70": 6,
    "70-80": 7, "80-90": 8, "90-100": 9
}

age_encoded = age_mapping[age_group]


if st.button("Predict Readmission Risk"):
    input_df = pd.DataFrame([{
        "age": age_encoded,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_medications": num_medications,
        "number_inpatient": number_inpatient
    }])

    prediction = model.predict(input_df)

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Hospital Readmission")
        st.write(
            "The patient has a **high probability of hospital readmission**. "
            "Enhanced monitoring and follow-up care are recommended."
        )
    else:
        st.success("✅ Low Risk of Hospital Readmission")
        st.write(
            "The patient has a **low probability of hospital readmission**. "
            "Standard discharge planning is sufficient."
        )

st.markdown("---")
st.caption("Machine Learning in Healthcare")
