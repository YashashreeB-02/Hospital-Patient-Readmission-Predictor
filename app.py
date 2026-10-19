import os
import joblib
import pandas as pd
import streamlit as st

from preprocess import prepare_single_input, AGE_ORDER
from chat_assistant import (
    ask_about_prediction,
    draft_followup_message,
    get_api_config,
    is_valid_api_key,
)

# Page configuration
st.set_page_config(
    page_title="Hospital Patient Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    .badge-low {
        display: inline-block;
        background: #10B98122;
        color: #10B981;
        border: 1px solid #10B98155;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
    }
    .badge-medium {
        display: inline-block;
        background: #F59E0B22;
        color: #F59E0B;
        border: 1px solid #F59E0B55;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
    }
    .badge-high {
        display: inline-block;
        background: #EF444422;
        color: #EF4444;
        border: 1px solid #EF444455;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
    }
    .chip-btn {
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .primary-btn>button {
        background: linear-gradient(135deg, #2563EB, #1D4ED8);
        color: white;
        font-size: 1.05rem;
        padding: 12px;
        border: none;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    .primary-btn>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45);
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .status-live {
        background: #10B9811A;
        color: #10B981;
        border: 1px solid #10B98144;
    }
    .status-offline {
        background: #6366F11A;
        color: #818CF8;
        border: 1px solid #6366F144;
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
        return {"model": artifact, "feature_columns": None, "encoders": None, "metrics": {}}


bundle = load_model_bundle()

# Initialize Session State
if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None
if "patient_context" not in st.session_state:
    st.session_state["patient_context"] = None
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []
if "followup_draft" not in st.session_state:
    st.session_state["followup_draft"] = ""

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hospital.png", width=60)
    st.title("Clinical AI System")
    st.caption("Hospital Patient Readmission Risk Stratification")

    st.markdown("---")
    st.subheader("Model Performance")
    if bundle and "metrics" in bundle and bundle["metrics"]:
        m = bundle["metrics"]
        st.metric("Test Accuracy", f"{m.get('test_accuracy', 0.6354) * 100:.1f}%")
        st.metric("ROC-AUC Score", f"{m.get('roc_auc', 0.6828):.3f}")
        st.metric("Trained Encounters", "99,343")
    else:
        st.info("Trained model loaded.")

    st.markdown("---")
    st.subheader("AI Copilot Settings")

    # Resolve API Key status
    default_key, default_model = get_api_config()
    is_live = is_valid_api_key(default_key)

    if is_live:
        st.markdown(
            '<div class="status-pill status-live">🟢 OpenAI Active (Live LLM)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-pill status-offline">💡 Educational Mode (Local Copilot)</div>',
            unsafe_allow_html=True,
        )
        st.caption("No valid API key detected. Running interactive local clinical rules mode.")

    with st.expander("⚙️ Configure API Key & Model", expanded=False):
        user_key_override = st.text_input(
            "OpenAI API Key (Session Override)",
            type="password",
            placeholder="sk-...",
            help="Your key is kept in memory during this session and never written to disk or Git.",
        )
        model_selection = st.selectbox(
            "Model Name",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
            help="OpenAI model for Talk to Your Data and Follow-up Draft generation.",
        )
        if user_key_override:
            st.session_state["custom_api_key"] = user_key_override.strip()
            st.session_state["custom_model"] = model_selection
            if is_valid_api_key(user_key_override):
                st.success("Custom session key applied!")
            else:
                st.warning("Key format appears invalid or is a placeholder.")

    st.markdown("---")
    st.markdown("""
    **Dataset:** UCI 130-US Hospitals (1999–2008)  
    **Algorithm:** HistGradientBoostingClassifier  
    **Prediction Target:** Any hospital readmission (recorded readmission vs. none)  
    **Clinical Boundary:** Academic & portfolio demonstration only. Not clinically validated.
    """)

# Main Header
st.title("🏥 Hospital Patient Readmission Predictor")
st.markdown(
    "Predict hospital readmission risk for diabetic inpatient encounters to support "
    "discharge planning, patient education, and clinical follow-up coordination."
)

with st.expander("ℹ️ Model Specifications & Clinical Limitations", expanded=False):
    st.markdown("""
    - **Dataset:** UCI Diabetes 130-US Hospitals (1999–2008), comprising 99,343 valid encounters after filtering out in-hospital mortalities.
    - **Target Variable Definition:** Binary classification of **any recorded readmission** (both `<30` days and `>30` days) versus no readmission (`NO`).
    - **Performance Metrics:** Test Accuracy: **63.54%** | ROC-AUC: **0.6828** (Evaluated on a 20% stratified test split).
    - **Clinical Disclaimer:** This tool is an educational portfolio prototype for clinical decision-support research. It is **not** approved by medical regulatory authorities as Software as a Medical Device (SaMD) and should not replace licensed professional judgment.
    """)

st.markdown("---")

if bundle is None:
    st.error("⚠️ Model file `readmission_model.pkl` not found. Please run `python train_model.py` to train the model.")
    st.stop()

# Input Layout
st.subheader("📋 Enter Patient & Encounter Data")

tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Demographics & Admission",
    "🩺 Clinical & Diagnoses",
    "📈 Prior Healthcare Visits",
    "💊 Medications & Lab Tests",
])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        age_group = st.selectbox(
            "Patient Age Group",
            AGE_ORDER,
            index=6,
            help="Patient age bracket at admission encounter.",
        )
        time_in_hospital = st.slider(
            "Days Spent in Hospital",
            min_value=1,
            max_value=14,
            value=4,
            help="Total length of stay for this encounter.",
        )
    with col2:
        admission_clean = st.selectbox(
            "Admission Type",
            ["Emergency", "Urgent", "Elective", "Other"],
            index=0,
            help="Urgency classification of the admission encounter.",
        )
        source_clean = st.selectbox(
            "Admission Source",
            ["Emergency Room", "Physician/Clinic Referral", "Transfer", "Other"],
            index=0,
            help="Patient referral source.",
        )
    with col3:
        discharge_clean = st.selectbox(
            "Discharge Destination",
            ["Home", "Home Health", "Transferred Facility", "AMA", "Other"],
            index=0,
            help="Discharge disposition (e.g. routine home discharge, home health services, or skilled facility).",
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
                "Other",
            ],
            index=0,
            help="Primary ICD-9 diagnosis grouping.",
        )
        number_diagnoses = st.slider(
            "Total Number of Diagnoses",
            min_value=1,
            max_value=16,
            value=8,
            help="Total recorded diagnoses on this encounter (comorbidity indicator).",
        )
    with col2:
        num_lab_procedures = st.slider(
            "Number of Lab Procedures Performed",
            min_value=1,
            max_value=100,
            value=42,
            help="Total diagnostic lab tests administered during stay.",
        )
    with col3:
        num_procedures = st.slider(
            "Number of Other Procedures",
            min_value=0,
            max_value=6,
            value=1,
            help="Surgical or diagnostic non-lab procedures performed.",
        )
        num_medications = st.slider(
            "Number of Medications Prescribed",
            min_value=1,
            max_value=50,
            value=14,
            help="Distinct medication prescriptions administered during stay.",
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
            help="Number of inpatient admissions in the preceding year.",
        )
    with col2:
        number_emergency = st.slider(
            "Prior Emergency Room Visits",
            min_value=0,
            max_value=10,
            value=0,
            help="Number of emergency visits in the preceding year.",
        )
    with col3:
        number_outpatient = st.slider(
            "Prior Outpatient Visits",
            min_value=0,
            max_value=20,
            value=0,
            help="Number of outpatient clinic visits in the preceding year.",
        )

with tab4:
    col1, col2 = st.columns(2)
    with col1:
        diabetesMed = st.selectbox(
            "Diabetes Medication Prescribed?",
            ["Yes", "No"],
            index=0,
            help="Was any diabetes medication prescribed during this encounter?",
        )
        change = st.selectbox(
            "Change in Medication Dosage/Type?",
            ["No", "Ch"],
            index=0,
            help="'Ch' indicates an active change or titration of diabetes medication.",
        )
    with col2:
        insulin = st.selectbox(
            "Insulin Regimen Status",
            ["No", "Steady", "Up", "Down"],
            index=1,
            help="'Up' indicates dosage increase; 'Down' indicates decrease; 'Steady' indicates continued regimen.",
        )
        A1Cresult = st.selectbox(
            "HbA1c Lab Test Result",
            ["None", "Norm", ">7", ">8"],
            index=0,
            help="HbA1c test result: None (not performed), Norm (<7%), >7%, or >8%.",
        )
        max_glu_serum = st.selectbox(
            "Max Glucose Serum Test Result",
            ["None", "Norm", ">200", ">300"],
            index=0,
            help="Blood glucose test threshold indicator.",
        )

st.markdown("---")

# Predict Button
btn_col1, btn_col2 = st.columns([3, 1])
with btn_col1:
    predict_clicked = st.button(
        "🚀 Analyze Readmission Risk",
        use_container_width=True,
        type="primary",
    )
with btn_col2:
    reset_clicked = st.button("🔄 Reset Assessment", use_container_width=True)

if reset_clicked:
    st.session_state["prediction_result"] = None
    st.session_state["patient_context"] = None
    st.session_state["chat_messages"] = []
    st.session_state["followup_draft"] = ""
    st.rerun()

# Run Inference when predict button is clicked
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
        'max_glu_serum': max_glu_serum,
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
    elif prob_readmit >= 0.42:
        risk_level = "Moderate Risk"
        badge_html = '<span class="badge-medium">🟡 MODERATE RISK OF READMISSION</span>'
    else:
        risk_level = "Low Risk"
        badge_html = '<span class="badge-low">🟢 LOW RISK OF READMISSION</span>'

    # Contributing Risk Factors Analysis
    factors = []
    if number_inpatient >= 2:
        factors.append(f"⚠️ **Frequent Hospitalizations:** {number_inpatient} prior inpatient admissions in preceding 12 months.")
    elif number_inpatient == 1:
        factors.append("⚠️ **Prior Inpatient Stay:** 1 hospitalization in preceding 12 months.")

    if number_emergency > 0:
        factors.append(f"⚠️ **Emergency Utilization:** {number_emergency} past emergency room encounters.")

    if number_diagnoses >= 8:
        factors.append(f"⚠️ **High Comorbidity Burden:** {number_diagnoses} simultaneous diagnoses recorded.")

    if time_in_hospital >= 7:
        factors.append(f"⚠️ **Extended Hospitalization:** {time_in_hospital} days in hospital (exceeds cohort average).")

    if insulin in ["Up", "Down"]:
        factors.append(f"⚠️ **Labile Glycemic Control:** Insulin dosage was actively adjusted ({insulin}).")

    if discharge_clean in ["Transferred Facility", "Home Health"]:
        factors.append(f"⚠️ **Post-Acute Care Required:** Discharged to {discharge_clean}.")

    if not factors:
        factors.append("✅ **No major high-risk warning flags detected.** General metabolic indicators are stable.")

    # Save to session state so user interactions do not cause assessment to disappear
    st.session_state["prediction_result"] = {
        "prob_readmit": prob_readmit,
        "risk_level": risk_level,
        "badge_html": badge_html,
        "factors": factors,
    }

    st.session_state["patient_context"] = {
        "prediction_target": (
            "Recorded hospital readmission (binary classification of any readmission "
            "vs. no readmission on the UCI Diabetes dataset; not a 30-day-specific prognosis)."
        ),
        "readmission_probability_percent": round(prob_readmit * 100, 1),
        "risk_category": risk_level,
        "patient_inputs": raw_input,
        "existing_rule_based_flags": factors,
        "verified_patient_level_model_explanation": None,
    }


# Render Prediction & AI Copilot Section whenever an assessment is available
if st.session_state.get("prediction_result") is not None:
    res = st.session_state["prediction_result"]
    prob_readmit = res["prob_readmit"]
    risk_level = res["risk_level"]
    badge_html = res["badge_html"]
    factors = res["factors"]
    patient_context = st.session_state["patient_context"]

    st.markdown("## 📊 Assessment Summary")
    st.markdown(badge_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Readmission Probability", f"{prob_readmit * 100:.1f}%")
    with col2:
        st.metric("Risk Classification", risk_level)
    with col3:
        st.metric("Cohort Baseline Delta", f"{(prob_readmit - 0.47) * 100:+.1f}% vs Avg")

    # Visual Meter
    st.markdown("**Risk Score Gauge:**")
    st.progress(min(1.0, max(0.0, prob_readmit)))

    # Risk Drivers and Clinical Protocol
    col_drivers, col_actions = st.columns(2)

    with col_drivers:
        st.markdown("### 🔍 Contributing Risk Factors")
        for factor in factors:
            st.markdown(f"- {factor}")

    with col_actions:
        st.markdown("### 📋 Recommended Discharge Protocol")
        if risk_level == "High Risk":
            st.markdown("""
            1. **Early Primary Care Follow-up:** Schedule in-person clinical follow-up within **5 to 7 days**.
            2. **Pharmacist Medication Reconciliation:** Conduct 1-on-1 review to mitigate adverse drug interactions.
            3. **Clinical Outreach Call:** Nurse telephone check-in at **48 hours** post-discharge.
            4. **Diabetes Education Consultation:** Ensure patient and caregiver understand insulin titration and home glucose monitoring.
            """)
        elif risk_level == "Moderate Risk":
            st.markdown("""
            1. **Outpatient Follow-up:** Schedule clinical check within **10 to 14 days**.
            2. **Medication Schedule:** Provide clear, written medication administration instructions.
            3. **Warning Signs Education:** Educate patient on early signs requiring medical attention.
            """)
        else:
            st.markdown("""
            1. **Standard Routine Follow-up:** Routine visit in **3 to 4 weeks**.
            2. **Discharge Packet:** Standard wellness, lifestyle, and dietary guidance provided.
            """)

    # AI Clinical Copilot Section
    st.markdown("---")
    st.subheader("💬 Talk to Your Data (AI Clinical Copilot)")
    st.caption(
        "Ask questions about this patient's prediction, risk factors, or follow-up plan. "
        "Supports multi-turn follow-up inquiries."
    )

    # Suggested Prompts (Patient-specific and dataset-level clinical inquiries)
    st.markdown("**Suggested Quick Inquiries:**")
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    selected_prompt = None
    with pcol1:
        if st.button("❓ Why this patient's risk?", key="prompt_why", use_container_width=True):
            selected_prompt = "Why was this risk level assigned to this patient?"
    with pcol2:
        if st.button("👥 Male vs. Female Risk?", key="prompt_gender", use_container_width=True):
            selected_prompt = "How many male and female patients are in the dataset and what are their readmission rates?"
    with pcol3:
        if st.button("🎂 High-Risk Age Groups?", key="prompt_age", use_container_width=True):
            selected_prompt = "Which age groups have the highest readmission rates?"
    with pcol4:
        if st.button("💊 Medication & Insulin?", key="prompt_meds", use_container_width=True):
            selected_prompt = "How do medications, insulin regimens, and dosage changes relate to readmission across the dataset?"

    # Display Existing Chat History
    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    chat_question = st.chat_input(
        "Ask about this prediction or ask a follow-up question...",
        key="patient_chat_input",
    )

    prompt_to_process = selected_prompt or chat_question

    if prompt_to_process:
        # Append User Message to session state
        st.session_state["chat_messages"].append({
            "role": "user",
            "content": prompt_to_process,
        })

        # Get API configuration
        custom_key = st.session_state.get("custom_api_key")
        custom_model = st.session_state.get("custom_model")

        # Generate Assistant Response
        try:
            with st.spinner("Analyzing encounter data..."):
                answer = ask_about_prediction(
                    patient_context,
                    prompt_to_process,
                    model_name=custom_model,
                    chat_history=st.session_state["chat_messages"][:-1],
                    api_key=custom_key,
                )
            st.session_state["chat_messages"].append({
                "role": "assistant",
                "content": answer,
            })
        except Exception as exc:
            err_msg = f"The AI assistant could not respond ({type(exc).__name__}: {exc}). Check your API configuration."
            st.session_state["chat_messages"].append({
                "role": "assistant",
                "content": err_msg,
            })

        st.rerun()

    if st.session_state["chat_messages"]:
        if st.button("🗑️ Clear Chat History", key="clear_chat_btn"):
            st.session_state["chat_messages"] = []
            st.rerun()

    # AI Follow-up Draft Generator
    st.markdown("---")
    st.subheader("✉️ AI Follow-up Draft Generator")
    st.caption(
        "Generate a structured, professional post-discharge message draft tailored to this encounter. "
        "Strict human-in-the-loop: drafts must be verified by clinical staff before use and are NEVER sent automatically."
    )

    draft_col1, draft_col2 = st.columns([2, 1])
    with draft_col1:
        gen_draft_clicked = st.button("📝 Generate Follow-up Draft", use_container_width=True)
    with draft_col2:
        if st.session_state.get("followup_draft"):
            if st.button("🗑️ Clear Draft", use_container_width=True):
                st.session_state["followup_draft"] = ""
                st.rerun()

    if gen_draft_clicked:
        custom_key = st.session_state.get("custom_api_key")
        custom_model = st.session_state.get("custom_model")
        try:
            with st.spinner("Drafting clinical follow-up communication..."):
                generated_draft = draft_followup_message(
                    patient_context,
                    model_name=custom_model,
                    api_key=custom_key,
                )
            st.session_state["followup_draft"] = generated_draft
            st.rerun()
        except Exception as exc:
            st.error(f"Could not generate draft ({type(exc).__name__}: {exc}). Check API configuration.")

    if st.session_state.get("followup_draft"):
        st.markdown("**Editable Message Draft (Review and edit before use):**")
        edited_draft = st.text_area(
            label="Editable Draft Content",
            value=st.session_state["followup_draft"],
            height=280,
            key="followup_draft_editor",
            help="Review all placeholders and clinical instructions carefully.",
        )
        st.session_state["followup_draft"] = edited_draft

        st.warning(
            "🔒 **Human-in-the-Loop Safeguard:** This message is an unverified draft generated for clinical coordination. "
            "No message has been sent. All details and patient identifiers must be verified by licensed staff before communication."
        )

st.markdown("---")
st.caption(
    "Hospital Patient Readmission Predictor • Academic & Engineering Portfolio Project • "
    "Machine Learning & Generative AI in Healthcare Systems"
)
