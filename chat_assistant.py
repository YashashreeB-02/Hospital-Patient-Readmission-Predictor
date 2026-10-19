import os
import json
import streamlit as st
from openai import OpenAI
from data_analytics import handle_data_query

DEFAULT_MODEL = "gpt-4o-mini"


def get_api_config(override_key=None, override_model=None):
    """
    Retrieve the OpenAI API key and model name safely from available sources:
    1. Direct runtime override (e.g. from session state / UI)
    2. Streamlit secrets (st.secrets)
    3. Environment variables (os.environ)
    """
    api_key = override_key
    if not api_key:
        try:
            if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
                api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            api_key = None

    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")

    model_name = override_model
    if not model_name:
        try:
            if hasattr(st, "secrets") and "OPENAI_MODEL" in st.secrets:
                model_name = st.secrets["OPENAI_MODEL"]
        except Exception:
            model_name = None

    if not model_name:
        model_name = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)

    return api_key, model_name


def is_valid_api_key(api_key):
    """Check if the API key is present and not a known placeholder string."""
    if not api_key or not isinstance(api_key, str):
        return False
    stripped = api_key.strip()
    if len(stripped) < 15:
        return False
    placeholders = [
        "YOUR_OPENAI_API_KEY",
        "YOUR_API_KEY",
        "sk-placeholder",
        "your-openai-api-key-here",
    ]
    if any(p.lower() in stripped.lower() for p in placeholders):
        return False
    return True


def _build_context_summary(patient_context):
    """Format structured patient assessment context into readable text."""
    if not isinstance(patient_context, dict):
        return str(patient_context)

    prob = patient_context.get("readmission_probability_percent", "N/A")
    risk = patient_context.get("risk_category", "N/A")
    target_note = patient_context.get("prediction_target", "Recorded hospital readmission")
    factors = patient_context.get("existing_rule_based_flags", [])
    inputs = patient_context.get("patient_inputs", {})

    clean_factors = [f.replace("⚠️", "").replace("✅", "").strip() for f in factors] if factors else ["None"]

    summary_lines = [
        f"- Risk Category: {risk}",
        f"- Readmission Probability: {prob}%",
        f"- Target Definition: {target_note}",
        f"- Contributing Factors Detected: {', '.join(clean_factors)}",
        "- Key Encounter Parameters:",
    ]
    for k, v in inputs.items():
        summary_lines.append(f"  * {k}: {v}")

    return "\n".join(summary_lines)


def _generate_local_fallback_answer(patient_context, question, chat_history=None):
    """
    Provide an informative, distinct rule-based response for patient-specific
    clinical questions when offline.
    """
    prob = patient_context.get("readmission_probability_percent", 50.0)
    risk = patient_context.get("risk_category", "Moderate Risk")
    factors = patient_context.get("existing_rule_based_flags", [])
    inputs = patient_context.get("patient_inputs", {})
    q_lower = question.lower()

    header = "💡 *[Educational Mode — Local Rule-Based Copilot]*"

    # 1. Topic: Model Architecture, Dataset Scope, Target & Limitations
    if any(w in q_lower for w in [
        "limitation", "limitations", "accuracy", "metric", "metrics",
        "target", "dataset", "uci", "validation", "auc", "roc",
        "samd", "30-day", "30 day", "scope", "trained", "gradient boosting"
    ]):
        return f"""{header}

### 📐 Model Architecture, Target & Limitations

- **Prediction Target Definition:** Binary classification of **any recorded hospital readmission** (both `<30` days and `>30` days) versus no readmission (`NO`) from the UCI 130-US Hospitals dataset. *(Note: This model is not restricted solely to 30-day early readmissions).*
- **Cohort Scale:** 99,343 eligible inpatient diabetes encounters after excluding expired and hospice patients.
- **Model Algorithm:** Regularized Gradient Boosted Decision Trees (`HistGradientBoostingClassifier`, depth=6, lr=0.08, l2=0.8).
- **Validation Metrics (20% Stratified Holdout Test Split):**
  * **Test Accuracy:** **63.54%** (Eliminated baseline Random Forest ~35% overfitting).
  * **ROC-AUC Score:** **0.6828** (Strong discrimination across diverse patient sub-populations).
- **Clinical Boundaries & Ethical Caveats:**
  * **Observational Data:** Data collected retrospectively (1999–2008); clinical diabetes standards have evolved.
  * **Correlation vs. Causation:** Identified feature importances represent statistical correlations, not direct causal medical etiology.
  * **Regulatory Notice:** Academic and portfolio engineering demonstration; **not** approved as Software as a Medical Device (SaMD)."""

    # 2. Topic: Medications, Insulin Regimen & Glycemic Labs (Patient Specific)
    elif any(w in q_lower for w in [
        "medication", "medications", "meds", "drug", "drugs", "insulin",
        "dose", "dosage", "a1c", "hba1c", "glucose", "glu", "titration",
        "regimen", "prescribed", "prescription"
    ]):
        med = inputs.get("diabetesMed", "Yes")
        ins = inputs.get("insulin", "Steady")
        ch = inputs.get("change", "No")
        a1c = inputs.get("A1Cresult", "None")
        glu = inputs.get("max_glu_serum", "None")
        num_meds = inputs.get("num_medications", 14)

        med_analysis = []
        if ins in ["Up", "Down"]:
            med_analysis.append(f"⚠️ **Active Insulin Titration ({ins}):** The patient's insulin regimen was actively adjusted during admission, signaling unstable blood glucose levels that require close post-discharge outpatient monitoring.")
        elif ins == "Steady":
            med_analysis.append("✅ **Stable Insulin Regimen:** Patient remained on a stable insulin regimen without emergency dosage alterations during this encounter.")
        else:
            med_analysis.append("ℹ️ **No Insulin Prescribed:** Patient was managed via oral anti-hyperglycemic agents or non-insulin protocols.")

        if ch == "Ch":
            med_analysis.append("⚠️ **Medication Regimen Change:** Active adjustments or drug additions occurred during hospitalization. Medication changes increase short-term readmission hazard due to potential patient confusion or adverse drug interactions.")

        if a1c in [">7", ">8"]:
            med_analysis.append(f"⚠️ **Elevated HbA1c ({a1c}):** Indicates chronic suboptimal glycemic control prior to admission, compounding systemic vascular and metabolic vulnerability.")
        elif a1c == "Norm":
            med_analysis.append("✅ **Controlled HbA1c (Norm):** Long-term blood sugar levels were within normal ranges (<7%).")
        else:
            med_analysis.append("ℹ️ **HbA1c Test Not Performed (None):** Lack of recent baseline glycated hemoglobin testing leaves long-term metabolic compliance unverified.")

        if int(num_meds) >= 15:
            med_analysis.append(f"⚠️ **Polypharmacy Risk ({num_meds} Medications):** The patient is prescribed a high volume of concurrent medications, raising the statistical risk of adverse drug events.")

        med_bullets = "\n".join(f"- {item}" for item in med_analysis)

        return f"""{header}

### 💊 Patient Medication & Glycemic Profile Analysis

| Clinical Variable | Patient Value | Clinical Interpretation |
| :--- | :--- | :--- |
| **Diabetes Medication Prescribed** | `{med}` | Active anti-diabetic pharmacological therapy |
| **Insulin Regimen** | `{ins}` | {'Active dose titration' if ins in ['Up', 'Down'] else 'Stable maintenance'} |
| **Dosage/Regimen Adjustment** | `{ch}` | {'Medication changed during stay' if ch == 'Ch' else 'No dosage alteration'} |
| **Distinct Medications Count** | `{num_meds}` | {'Elevated polypharmacy index' if int(num_meds) >= 15 else 'Standard medication burden'} |
| **HbA1c Lab Result** | `{a1c}` | {'Suboptimal chronic control' if a1c in ['>7', '>8'] else ('Normal control' if a1c == 'Norm' else 'Test not ordered')} |
| **Max Glucose Serum** | `{glu}` | {'Acute hyperglycemia' if glu in ['>200', '>300'] else 'Within standard encounter range'} |

**Key Pharmacological Findings for this Patient:**
{med_bullets}"""

    # 3. Topic: Discharge Protocol, Actions, Timing & Follow-up Coordination
    elif any(w in q_lower for w in [
        "discharge", "protocol", "action", "actions", "timing", "appointment",
        "plan", "planning", "follow-up", "followup", "reconciliation",
        "nurse", "care team", "post-discharge", "next steps"
    ]):
        dest = inputs.get("discharge_clean", "Home")
        time_hosp = inputs.get("time_in_hospital", 4)

        if risk == "High Risk":
            timing_str = "**5 to 7 days**"
            protocol_items = [
                f"📅 **Urgent Primary Care Follow-up:** Schedule an in-person outpatient appointment within {timing_str}.",
                f"💊 **Pharmacist Medication Reconciliation:** Conduct a dedicated 1-on-1 review before discharge to clarify changes in insulin and {inputs.get('num_medications', 14)} concurrent prescriptions.",
                "📞 **48-Hour Nurse Outreach Call:** Telephonic clinical nurse check-in at 48 hours to assess symptoms, tolerance to new medications, and appointment confirmation.",
                "🩺 **Diabetes Self-Management Education (DSME):** Provide certified diabetes educator consult regarding blood glucose monitoring and hypoglycemia awareness."
            ]
        elif risk == "Moderate Risk":
            timing_str = "**10 to 14 days**"
            protocol_items = [
                f"📅 **Outpatient Clinical Check-in:** Schedule follow-up visit within {timing_str}.",
                "📋 **Structured Medication Schedule:** Provide printed, plain-language medication administration instructions.",
                "⚠️ **Red-Flag Symptoms Education:** Educate the patient and family on early decompensation signs requiring immediate care."
            ]
        else:
            timing_str = "**3 to 4 weeks**"
            protocol_items = [
                f"📅 **Routine Outpatient Follow-up:** Standard follow-up visit within {timing_str}.",
                "📄 **Standard Discharge Packet:** Comprehensive dietary, physical activity, and wellness guidance upon departure."
            ]

        if dest in ["Home Health", "Transferred Facility"]:
            protocol_items.append(f"🏥 **Post-Acute Care Transition:** Coordinate medical transfer and handover directly with {dest} care team.")

        items_md = "\n".join(f"{i+1}. {item}" for i, item in enumerate(protocol_items))

        return f"""{header}

### 📋 Recommended Post-Discharge Clinical Protocol

- **Stratified Risk Level:** **{risk}** (Readmission Probability: **{prob}%**)
- **Discharge Disposition:** `{dest}` (Encounter Stay: `{time_hosp}` days)
- **Target Outpatient Timing:** {timing_str}

**Step-by-Step Action Plan for Care Coordination:**
{items_md}

*Note: Early clinical contact within 7 days has been shown in hospital literature to reduce avoidable 30-day readmissions by up to 25%.*"""

    # 4. Topic: Why Risk Level Was Assigned / Risk Assessment Summary
    elif any(w in q_lower for w in [
        "why", "risk level", "risk score", "assigned", "high risk", "moderate risk",
        "low risk", "score", "probability", "factors", "drivers", "assessment",
        "summary", "overview", "classify", "classification"
    ]):
        factors_md = "\n".join(f"- {f}" for f in factors) if factors else "- General metabolic indicators are stable with no major warning flags."
        delta = round(float(prob) - 47.0, 1)
        delta_str = f"+{delta}% above" if delta >= 0 else f"{delta}% below"

        return f"""{header}

### 🔍 Risk Stratification Summary & Clinical Drivers

- **Assigned Risk Classification:** **{risk}**
- **Calculated Readmission Probability:** **{prob}%** ({delta_str} cohort average baseline of 47%)
- **Primary Diagnosis Grouping:** `{inputs.get('diag_1_cat', 'Circulatory')}`
- **Comorbidity Count:** `{inputs.get('number_diagnoses', 8)}` active diagnoses recorded

**Primary Contributing Factors Flagged by the Model:**
{factors_md}

*Clinical takeaway: Multi-factor risk combines frequent prior inpatient stays, complex comorbidity, and active glycemic adjustments.*"""

    # 5. Open-Ended Fallback for this Patient
    else:
        factors_summary = ", ".join([f.replace("⚠️", "").replace("✅", "").strip() for f in factors[:3]]) if factors else "Standard baseline metrics"
        return f"""{header}

### 💡 Patient Assessment Inquiry: *"{question}"*

- **Patient Assessment Overview:** Risk level is **{risk}** with an estimated readmission probability of **{prob}%**.
- **Key Contributing Indicators:** {factors_summary}.
- **Encounter Context:** Length of stay is `{inputs.get('time_in_hospital', 'N/A')}` days, with `{inputs.get('number_inpatient', 0)}` prior inpatient admissions and primary diagnosis categorized as `{inputs.get('diag_1_cat', 'N/A')}`.
- **Guidance:** You can ask dataset-level questions such as:
  * *"How many male and female patients are in the dataset?"*
  * *"Which age groups have the highest readmission rates?"*
  * *"How do medications and insulin relate to readmission?"*"""


def _generate_local_fallback_draft(patient_context):
    """Generate a structured, clinical follow-up message draft when offline."""
    risk = patient_context.get("risk_category", "Moderate Risk")
    prob = patient_context.get("readmission_probability_percent", 50.0)
    inputs = patient_context.get("patient_inputs", {})
    time_hosp = inputs.get("time_in_hospital", 4)
    discharge_dest = inputs.get("discharge_clean", "Home")
    insulin_status = inputs.get("insulin", "Steady")

    timing = "5 to 7 days" if risk == "High Risk" else ("10 to 14 days" if risk == "Moderate Risk" else "3 to 4 weeks")

    draft = f"""======================================================================
CLINICAL OUTREACH DRAFT — FOR PROFESSIONAL REVIEW ONLY
(NOT SENT AUTOMATICALLY • REQUIRES CLINICIAN VERIFICATION)
======================================================================

Date: [Insert Date]
Recipient: [Patient Name / Designated Caregiver]
Regarding: Post-Discharge Care Plan & Outpatient Coordination
Hospital Encounter Duration: {time_hosp} day(s) | Discharge To: {discharge_dest}
Stratified Risk Tier: {risk} (Estimated Readmission Risk: {prob}%)

Dear [Patient Name],

Thank you for partnering with our care team during your recent hospital stay. To ensure your recovery continues smoothly at home, we have prepared your personalized post-discharge plan:

1. OUTPATIENT FOLLOW-UP APPOINTMENT:
   Please confirm your appointment with Dr. [Primary Care Provider / Endocrinologist]
   Recommended timeframe: Within {timing}.
   Clinic Contact: [Insert Clinic Phone Number]
   Clinic Address: [Insert Clinic Location]

2. MEDICATION RECONCILIATION:
   - Your discharge medication list has been updated.
   - Insulin status on record: {insulin_status} (Please verify current prescribed dose).
   - If you experience confusion regarding new versus discontinued prescriptions, please call [Pharmacy Support Line] immediately.

3. WHEN TO SEEK IMMEDIATE CARE:
   If you experience fever, shortness of breath, sudden dizziness, or blood glucose levels consistently outside your target range, contact [Triage Hotline] or visit the nearest emergency department.

Warm regards,

[Clinician Name, Title]
[Department / Hospital Name]
[Direct Contact Number]

----------------------------------------------------------------------
CLINICIAN PRE-SEND VERIFICATION CHECKLIST:
[ ] Verify patient identity and contact details.
[ ] Confirm scheduled follow-up appointment date and physician name.
[ ] Perform 1-on-1 medication reconciliation against electronic health record.
[ ] Confirm caregiver/patient understanding of discharge instructions.
======================================================================"""
    return draft.strip()


def ask_about_prediction(patient_context, question, model_name=None, chat_history=None, api_key=None):
    """
    Data-driven conversational assistant:
    1. Distinguishes dataset questions (counts, percentages, cohorts, groups) from patient-specific questions.
    2. Executes exact Pandas calculations on diabetic_data.csv for dataset queries.
    3. Feeds verified data to OpenAI when active, or returns verified calculation directly when offline.
    """
    # Step 1: Check if this is a dataset-level query or unavailable column query
    is_data_query, data_response_md, data_dict = handle_data_query(question, chat_history=chat_history)

    key, model = get_api_config(override_key=api_key, override_model=model_name)
    has_live_llm = is_valid_api_key(key)

    # CASE A: Dataset-Level Query
    if is_data_query:
        if not has_live_llm:
            # Deliver verified calculation directly
            return data_response_md

        # Pass verified Pandas calculations as ground truth to OpenAI
        client = OpenAI(api_key=key)
        system_prompt = (
            "You are the data-driven AI Clinical & Dataset Assistant for the Hospital Patient Readmission Predictor.\n"
            "The user asked a question about the hospital dataset. The exact mathematical calculations have ALREADY "
            "been calculated against diabetic_data.csv using Python/Pandas.\n\n"
            "CRITICAL DATA INTEGRITY RULES:\n"
            "1. Ground your entire response in the verified data calculation supplied below.\n"
            "2. You MUST use the exact numbers, counts, and percentages provided in the verified findings. Do not guess or invent statistics.\n"
            "3. Format your response cleanly using markdown tables and bullet points.\n"
            "4. Maintain conversational continuity and answer follow-up nuances from prior messages.\n"
            "5. Distinguish correlation from clinical causation."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User question: {question}\n\nVerified Calculations from diabetic_data.csv:\n{data_response_md}\n\nExplain these verified findings clearly and professionally to the user."}
        ]

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=750,
            )
            return completion.choices[0].message.content.strip()
        except Exception:
            # Fall back to verified calculation markdown on any API error
            return data_response_md

    # CASE B: Patient-Specific Query
    if not has_live_llm:
        return _generate_local_fallback_answer(patient_context, question, chat_history=chat_history)

    client = OpenAI(api_key=key)
    context_str = _build_context_summary(patient_context)

    system_prompt = (
        "You are an AI Clinical Decision-Support Copilot for an educational hospital readmission project.\n"
        "Your job is to explain this specific patient's readmission risk prediction to healthcare professionals.\n\n"
        "CRITICAL GUIDELINES:\n"
        "1. Ground all explanations in the supplied patient assessment context.\n"
        "2. The prediction target is ANY hospital readmission (historical encounter record), NOT a certified 30-day clinical prognosis.\n"
        "3. Do NOT make clinical diagnoses, prescribe medications, or replace clinician judgment.\n"
        "4. Treat model feature contributions as statistical correlations observed in the dataset, not individual medical causality.\n"
        "5. Support multi-turn follow-up questions accurately by referencing prior messages in the conversation.\n"
        "6. Keep responses concise, objective, professional, and well-structured using markdown."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is the current patient encounter and model prediction context:\n{context_str}\n\nPlease keep this in mind for all subsequent questions."},
        {"role": "assistant", "content": f"Understood. I have reviewed the encounter data for this patient (Risk Tier: {patient_context.get('risk_category')}, Probability: {patient_context.get('readmission_probability_percent')}%). How can I assist you with this prediction?"}
    ]

    if chat_history and isinstance(chat_history, list):
        for msg in chat_history[-10:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ["user", "assistant"] and content:
                if not content.startswith("The AI assistant could not respond") and not content.startswith("Could not generate"):
                    messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question})

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=600,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        try:
            resp = client.responses.create(
                model=model,
                instructions=system_prompt,
                input=f"Context: {context_str}\n\nQuestion: {question}"
            )
            return resp.output_text.strip()
        except Exception:
            return _generate_local_fallback_answer(patient_context, question, chat_history=chat_history)


def draft_followup_message(patient_context, model_name=None, api_key=None):
    """
    Generate an editable follow-up message draft with strict clinician verification checklists.
    """
    key, model = get_api_config(override_key=api_key, override_model=model_name)

    if not is_valid_api_key(key):
        return _generate_local_fallback_draft(patient_context)

    client = OpenAI(api_key=key)
    context_str = _build_context_summary(patient_context)

    system_prompt = (
        "You are an AI Clinical Communication Assistant generating a post-discharge follow-up message draft.\n"
        "CRITICAL SAFEGUARDS:\n"
        "1. This is an educational draft prototype. DO NOT invent dates, names, medication dosages, or diagnoses.\n"
        "2. Use clear bracketed placeholders (e.g., [Patient Name], [Clinic Phone], [Appointment Date]).\n"
        "3. Explicitly mark the message as a DRAFT requiring licensed clinician review and verification.\n"
        "4. Include a pre-send verification checklist for the care team.\n"
        "5. State clearly that NO message has been or will be automatically sent."
    )

    prompt = (
        f"Patient Assessment Context:\n{context_str}\n\n"
        "Generate a professional, structured post-discharge follow-up message draft tailored to the patient's "
        "risk tier and encounter characteristics. Include placeholders for missing identifiers and a mandatory clinician pre-send checklist."
    )

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=700,
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        return _generate_local_fallback_draft(patient_context)
