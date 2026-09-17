import os
import joblib
import pandas as pd
import streamlit as st
from preprocess import prepare_single_input, AGE_ORDER

# Page configuration
st.set_page_config(
    page_title="Hospital Patient Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .badge-low {
        display: inline-block;
        background: #10B98122;
        color: #10B981;
        border: 1px solid #10B98155;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-medium {
        display: inline-block;
        background: #F59E0B22;
        color: #F59E0B;
        border: 1px solid #F59E0B55;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-high {
        display: inline-block;
        background: #EF444422;
        color: #EF4444;
        border: 1px solid #EF444455;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #2563EB, #1D4ED8);
        color: white;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 12px;
        border-radius: 10px;
        border: none;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(37,99,235,0.4);
    }
</style>
""", unsafe_allow_html=True)

# Load model bundle
@st.cache_resource
def load_model_bundle():
    model_path = "readmission_model.pkl"
    if not os.path.exists(model_path):
        return None
    artifact = joblib.load(model_path)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact
    else:
        # Fallback if bare model
        return {"model": artifact, "feature_columns": None, "encoders": None, "metrics": {}}

bundle = load_model_bundle()

# Sidebar: System & Model Information
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hospital.png", width=64)
    st.title("Clinical AI Assistant")
    st.caption("Hospital Readmission Risk Stratification")

    st.markdown("---")
    st.subheader("Model Performance")
    if bundle and "metrics" in bundle and bundle["metrics"]:
        m = bundle["metrics"]
        st.metric("Test Accuracy", f"{m.get('test_accuracy', 0.635) * 100:.1f}%")
        st.metric("ROC-AUC Score", f"{m.get('roc_auc', 0.683):.3f}")
        st.metric("Trained Encounters", "99,343")
    else:
        st.info("Trained model active.")

    st.markdown("---")
    st.markdown("""
    **Dataset:** UCI 130-US Hospitals  
    **Algorithm:** Regularized Gradient Boosting  
    **Clinical Goal:** Early intervention for high-risk patients before discharge.
    """)

# Main Header
st.title("🏥 Hospital Patient Readmission Predictor")
st.markdown("Predict the 30-day readmission risk for diabetic inpatient admissions to support discharge planning and clinical interventions.")
st.markdown("---")

if bundle is None:
    st.error("⚠️ Model file `readmission_model.pkl` not found. Please run `python train_model.py` first to train the model.")
    st.stop()

# Input Layout
st.subheader("📋 Enter Patient & Encounter Data")

tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Demographics & Admission",
    "🩺 Clinical & Diagnoses",
    "📈 Prior Healthcare Visits",
    "💊 Medications & Lab Tests"
])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        age_group = st.selectbox(
            "Patient Age Group",
            AGE_ORDER,
            index=6,
            help="Patient age range at the time of hospital admission."
        )
        time_in_hospital = st.slider(
            "Days Spent in Hospital",
            min_value=1,
            max_value=14,
            value=4,
            help="Total length of stay for this encounter."
        )
    with col2:
        admission_clean = st.selectbox(
            "Admission Type",
            ["Emergency", "Urgent", "Elective", "Other"],
            index=0,
            help="Urgency classification of the admission."
        )
        source_clean = st.selectbox(
            "Admission Source",
            ["Emergency Room", "Physician/Clinic Referral", "Transfer", "Other"],
            index=0,
            help="Where the patient was referred from."
        )
    with col3:
        discharge_clean = st.selectbox(
            "Discharge Destination",
            ["Home", "Home Health", "Transferred Facility", "AMA", "Other"],
            index=0,
            help="Where the patient was discharged (e.g., routine home vs. home health care vs. skilled facility)."
        )

with tab2:
    col1, col2, col3 = st.columns(3)
    with col1:
        diag_1_cat = st.selectbox(
            "Primary Diagnosis Category",
            [
                "Circulatory",
                "Respiratory",
                "Digestive",
                "Diabetes",
                "Injury",
                "Musculoskeletal",
                "Genitourinary",
                "Neoplasms",
                "Other"
            ],
            index=0,
            help="Primary ICD-9 diagnosis category assigned to this encounter."
        )
        number_diagnoses = st.slider(
            "Total Number of Diagnoses",
            min_value=1,
            max_value=16,
            value=8,
            help="Total number of diagnoses coded on the patient's record (indicates comorbidity)."
        )
    with col2:
        num_lab_procedures = st.slider(
            "Number of Lab Procedures Performed",
            min_value=1,
            max_value=100,
            value=42,
            help="Total laboratory tests administered during hospital stay."
        )
    with col3:
        num_procedures = st.slider(
            "Number of Other Procedures",
            min_value=0,
            max_value=6,
            value=1,
            help="Surgical or diagnostic procedures other than lab tests."
        )
        num_medications = st.slider(
            "Number of Medications Prescribed",
            min_value=1,
            max_value=50,
            value=14,
            help="Distinct medication prescriptions administered during stay."
        )

with tab3:
    st.caption("Prior hospital healthcare utilization in the 12 months preceding this encounter:")
    col1, col2, col3 = st.columns(3)
    with col1:
        number_inpatient = st.slider(
            "Prior Inpatient Visits",
            min_value=0,
            max_value=10,
            value=1,
            help="Number of inpatient admissions in the preceding year."
        )
    with col2:
        number_emergency = st.slider(
            "Prior Emergency Room Visits",
            min_value=0,
            max_value=10,
            value=0,
            help="Number of emergency visits in the preceding year."
        )
    with col3:
        number_outpatient = st.slider(
            "Prior Outpatient Visits",
            min_value=0,
            max_value=20,
            value=0,
            help="Number of outpatient clinic visits in the preceding year."
        )

with tab4:
    col1, col2 = st.columns(2)
    with col1:
        diabetesMed = st.selectbox(
            "Diabetes Medication Prescribed?",
            ["Yes", "No"],
            index=0,
            help="Was any diabetes medication prescribed during this encounter?"
        )
        change = st.selectbox(
            "Change in Medication Dosage/Type?",
            ["No", "Ch"],
            index=0,
            help="'Ch' indicates a change in diabetes medication dosage or initiation of a new drug."
        )
    with col2:
        insulin = st.selectbox(
            "Insulin Regimen Status",
            ["No", "Steady", "Up", "Down"],
            index=1,
            help="'Up' indicates dosage was increased; 'Down' indicates decreased; 'Steady' indicates continued."
        )
        A1Cresult = st.selectbox(
            "HbA1c Lab Test Result",
            ["None", "Norm", ">7", ">8"],
            index=0,
            help="HbA1c test status: None (not performed), Norm (<7%), >7%, or >8%."
        )
        max_glu_serum = st.selectbox(
            "Max Glucose Serum Test Result",
            ["None", "Norm", ">200", ">300"],
            index=0,
            help="Blood glucose test result threshold."
        )

st.markdown("---")

# Prediction Trigger
predict_clicked = st.button("🚀 Analyze Readmission Risk", use_container_width=True)

if predict_clicked:
    raw_input = {
        'age': age_group,
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_outpatient': number_outpatient,
        'number_emergency': number_emergency,
        'number_inpatient': number_inpatient,
        'number_diagnoses': number_diagnoses,
        'diag_1_cat': diag_1_cat,
        'discharge_clean': discharge_clean,
        'admission_clean': admission_clean,
        'source_clean': source_clean,
        'diabetesMed': diabetesMed,
        'change': change,
        'insulin': insulin,
        'A1Cresult': A1Cresult,
        'max_glu_serum': max_glu_serum
    }

    feature_cols = bundle["feature_columns"]
    encoders = bundle["encoders"]
    model = bundle["model"]

    X_processed = prepare_single_input(raw_input, feature_cols, encoders)

    # Inference
    prediction = int(model.predict(X_processed)[0])
    probabilities = model.predict_proba(X_processed)[0]
    prob_readmit = float(probabilities[1])

    # Determine risk category
    if prob_readmit >= 0.55:
        risk_level = "High Risk"
        badge_html = '<span class="badge-high">🔴 HIGH RISK OF READMISSION</span>'
        banner_type = "error"
    elif prob_readmit >= 0.42:
        risk_level = "Moderate Risk"
        badge_html = '<span class="badge-medium">🟡 MODERATE RISK OF READMISSION</span>'
        banner_type = "warning"
    else:
        risk_level = "Low Risk"
        badge_html = '<span class="badge-low">🟢 LOW RISK OF READMISSION</span>'
        banner_type = "success"

    st.markdown("## 📊 Assessment Summary")
    st.markdown(badge_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Readmission Probability", f"{prob_readmit * 100:.1f}%")
    with col2:
        st.metric("Risk Classification", risk_level)
    with col3:
        st.metric("Baseline Risk Comparison", f"{(prob_readmit - 0.47) * 100:+.1f}% vs Avg")

    # Visual Meter
    st.markdown("**Risk Score Gauge:**")
    st.progress(min(1.0, max(0.0, prob_readmit)))

    # Risk Drivers and Clinical Protocol
    col_drivers, col_actions = st.columns(2)

    with col_drivers:
        st.markdown("### 🔍 Contributing Risk Factors")
        factors = []
        if number_inpatient >= 2:
            factors.append(f"⚠️ **Frequent Hospitalizations:** Patient has {number_inpatient} prior inpatient admissions.")
        elif number_inpatient == 1:
            factors.append("⚠️ **Prior Inpatient Stay:** 1 hospitalization in preceding 12 months.")

        if number_emergency > 0:
            factors.append(f"⚠️ **Emergency Utilization:** {number_emergency} past emergency room encounters.")

        if number_diagnoses >= 8:
            factors.append(f"⚠️ **High Comorbidity Burden:** {number_diagnoses} simultaneous diagnoses recorded.")

        if time_in_hospital >= 7:
            factors.append(f"⚠️ **Extended Stay:** {time_in_hospital} days in hospital (exceeds average).")

        if insulin in ["Up", "Down"]:
            factors.append(f"⚠️ **Unstable Glycemic Control:** Insulin dosage was actively adjusted ({insulin}).")

        if discharge_clean in ["Transferred Facility", "Home Health"]:
            factors.append(f"⚠️ **Post-Acute Care Required:** Discharged to {discharge_clean}.")

        if not factors:
            factors.append("✅ **No major high-risk warning flags detected.** General metabolic indicators are stable.")

        for factor in factors:
            st.markdown(f"- {factor}")

    with col_actions:
        st.markdown("### 📋 Recommended Discharge Protocol")
        if risk_level == "High Risk":
            st.markdown("""
            1. **Post-Discharge Appointment:** Schedule in-person primary care follow-up within **5 to 7 days**.
            2. **Pharmacist Medication Reconciliation:** Conduct 1-on-1 review to prevent adverse drug events.
            3. **Telephone Check-In:** Clinical nurse call at **48 hours** post-discharge.
            4. **Diabetes Education Consult:** Ensure patient understands insulin adjustments and home glucose monitoring.
            """)
        elif risk_level == "Moderate Risk":
            st.markdown("""
            1. **Outpatient Follow-up:** Schedule follow-up visit within **10 to 14 days**.
            2. **Medication Instructions:** Provide written instructions for medication schedule.
            3. **Symptom Awareness:** Educate patient on early warning signs requiring emergency care.
            """)
        else:
            st.markdown("""
            1. **Standard Routine Follow-up:** Routine visit in **3 to 4 weeks**.
            2. **Standard Discharge Packet:** Standard wellness and diet guidance provided upon departure.
            """)

st.markdown("---")
st.caption("Hospital Patient Readmission Predictor • Academic Capstone Project • Machine Learning in Healthcare")
