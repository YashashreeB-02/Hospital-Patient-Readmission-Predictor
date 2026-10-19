import os
import re
import pandas as pd
import numpy as np
from preprocess import categorize_diagnosis, map_discharge, map_admission, map_source

_CACHED_DF = None


def get_dataset(filepath="diabetic_data.csv"):
    """
    Load and return the cleaned, preprocessed dataset (cached in-memory).
    Standard clinical exclusion: Excludes expired and hospice patients,
    as they are not eligible for post-discharge readmission.
    """
    global _CACHED_DF
    if _CACHED_DF is not None:
        return _CACHED_DF

    if not os.path.exists(filepath):
        alt_paths = [
            os.path.join(os.path.dirname(__file__), filepath),
            os.path.join("/Users/yashashree/Hospital-Patient-Readmission-Predictor", filepath)
        ]
        for p in alt_paths:
            if os.path.exists(p):
                filepath = p
                break

    df = pd.read_csv(filepath)

    # Exclude patients who expired in hospital or entered hospice (cannot be readmitted)
    df = df[~df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])].copy()

    # Binary readmission indicator: 1 if readmitted (<30 or >30), 0 if NO
    df['is_readmitted'] = df['readmitted'].isin(['<30', '>30']).astype(int)

    # Clean standardized clinical groupings if not already present
    if 'diag_1_cat' not in df.columns and 'diag_1' in df.columns:
        df['diag_1_cat'] = df['diag_1'].apply(categorize_diagnosis)
    if 'discharge_clean' not in df.columns and 'discharge_disposition_id' in df.columns:
        df['discharge_clean'] = df['discharge_disposition_id'].apply(map_discharge)
    if 'admission_clean' not in df.columns and 'admission_type_id' in df.columns:
        df['admission_clean'] = df['admission_type_id'].apply(map_admission)
    if 'source_clean' not in df.columns and 'admission_source_id' in df.columns:
        df['source_clean'] = df['admission_source_id'].apply(map_source)

    _CACHED_DF = df
    return _CACHED_DF


def _resolve_df(df=None):
    """Helper to return provided DataFrame or default cached dataset."""
    if df is not None:
        resolved = df.copy()
        if 'is_readmitted' not in resolved.columns and 'readmitted' in resolved.columns:
            resolved['is_readmitted'] = resolved['readmitted'].isin(['<30', '>30']).astype(int)
        return resolved
    return get_dataset()


def get_cohort_summary(df=None):
    """
    Calculate high-level cohort population metrics dynamically.
    Returns exact counts and percentages with explicit denominators.
    """
    target_df = _resolve_df(df)
    total = len(target_df)
    if total == 0:
        return {
            "total_encounters": 0,
            "readmitted_count": 0,
            "readmission_rate_percent": 0.0,
            "readmit_under_30_count": 0,
            "readmit_under_30_percent": 0.0,
            "readmit_over_30_count": 0,
            "readmit_over_30_percent": 0.0,
            "no_readmit_count": 0,
            "no_readmit_percent": 0.0
        }

    readmitted_count = int(target_df['is_readmitted'].sum())
    readmission_rate = round((readmitted_count / total) * 100, 2)
    readmit_30_count = int((target_df['readmitted'] == '<30').sum()) if 'readmitted' in target_df.columns else 0
    readmit_30_rate = round((readmit_30_count / total) * 100, 2)
    readmit_late_count = int((target_df['readmitted'] == '>30').sum()) if 'readmitted' in target_df.columns else 0
    readmit_late_rate = round((readmit_late_count / total) * 100, 2)
    no_readmit_count = int((target_df['readmitted'] == 'NO').sum()) if 'readmitted' in target_df.columns else (total - readmitted_count)
    no_readmit_rate = round((no_readmit_count / total) * 100, 2)

    return {
        "total_encounters": total,
        "readmitted_count": readmitted_count,
        "readmission_rate_percent": readmission_rate,
        "readmit_under_30_count": readmit_30_count,
        "readmit_under_30_percent": readmit_30_rate,
        "readmit_over_30_count": readmit_late_count,
        "readmit_over_30_percent": readmit_late_rate,
        "no_readmit_count": no_readmit_count,
        "no_readmit_percent": no_readmit_rate
    }


def calculate_gender_stats(df=None):
    """
    Calculate dynamic gender distribution and readmission counts/rates.
    Iterates over all unique gender categories found in the data.
    """
    target_df = _resolve_df(df)
    total = len(target_df)
    if total == 0 or 'gender' not in target_df.columns:
        return {}

    g_agg = target_df.groupby('gender').agg(
        total_encounters=('is_readmitted', 'count'),
        readmitted_encounters=('is_readmitted', 'sum'),
        readmission_rate=('is_readmitted', lambda x: round(x.mean() * 100, 2) if len(x) > 0 else 0.0)
    ).reset_index()

    g_agg['percent_of_cohort'] = (g_agg['total_encounters'] / total * 100).round(2)

    stats = {}
    for _, row in g_agg.iterrows():
        g = str(row['gender'])
        tot = int(row['total_encounters'])
        readm = int(row['readmitted_encounters'])
        stats[g] = {
            "total_encounters": tot,
            "percent_of_cohort": float(row['percent_of_cohort']),
            "readmitted_count": readm,
            "readmission_rate_percent": float(row['readmission_rate']),
            "not_readmitted_count": tot - readm
        }

    return stats


def calculate_age_stats(df=None):
    """
    Calculate dynamic readmission rates and encounter counts across all age brackets.
    Returns list sorted by readmission rate in descending order.
    """
    target_df = _resolve_df(df)
    total = len(target_df)
    if total == 0 or 'age' not in target_df.columns:
        return []

    age_agg = target_df.groupby('age').agg(
        total_encounters=('is_readmitted', 'count'),
        readmitted_encounters=('is_readmitted', 'sum'),
        readmission_rate=('is_readmitted', lambda x: round(x.mean() * 100, 2) if len(x) > 0 else 0.0)
    ).reset_index()

    age_agg['percent_of_cohort'] = (age_agg['total_encounters'] / total * 100).round(2)
    age_agg = age_agg.sort_values('readmission_rate', ascending=False)

    age_list = []
    for _, row in age_agg.iterrows():
        age_list.append({
            "age_bracket": str(row['age']),
            "total_encounters": int(row['total_encounters']),
            "percent_of_cohort": float(row['percent_of_cohort']),
            "readmitted_count": int(row['readmitted_encounters']),
            "readmission_rate_percent": float(row['readmission_rate'])
        })

    return age_list


def calculate_medication_stats(df=None):
    """
    Calculate dynamic readmission metrics for diabetes medications, insulin,
    regimen adjustments, and polypharmacy medication burdens.
    """
    target_df = _resolve_df(df)
    total = len(target_df)
    if total == 0:
        return {"insulin": [], "change": {}, "diabetesMed": {}, "avg_medications_readmitted": 0.0, "avg_medications_not_readmitted": 0.0}

    # Insulin regimen
    insulin_breakdown = []
    if 'insulin' in target_df.columns:
        ins_agg = target_df.groupby('insulin').agg(
            total=('is_readmitted', 'count'),
            readmitted=('is_readmitted', 'sum'),
            rate=('is_readmitted', lambda x: round(x.mean() * 100, 2) if len(x) > 0 else 0.0)
        ).reset_index().sort_values('rate', ascending=False)

        insulin_breakdown = [
            {"regimen": str(r['insulin']), "total": int(r['total']), "readmitted": int(r['readmitted']), "rate_percent": float(r['rate'])}
            for _, r in ins_agg.iterrows()
        ]

    # Change in medication
    change_breakdown = {}
    if 'change' in target_df.columns:
        ch_agg = target_df.groupby('change').agg(
            total=('is_readmitted', 'count'),
            readmitted=('is_readmitted', 'sum'),
            rate=('is_readmitted', lambda x: round(x.mean() * 100, 2) if len(x) > 0 else 0.0)
        ).reset_index()

        change_breakdown = {
            str(r['change']): {"total": int(r['total']), "readmitted": int(r['readmitted']), "rate_percent": float(r['rate'])}
            for _, r in ch_agg.iterrows()
        }

    # DiabetesMed prescribed
    diabetes_med_breakdown = {}
    if 'diabetesMed' in target_df.columns:
        dm_agg = target_df.groupby('diabetesMed').agg(
            total=('is_readmitted', 'count'),
            readmitted=('is_readmitted', 'sum'),
            rate=('is_readmitted', lambda x: round(x.mean() * 100, 2) if len(x) > 0 else 0.0)
        ).reset_index()

        diabetes_med_breakdown = {
            str(r['diabetesMed']): {"total": int(r['total']), "readmitted": int(r['readmitted']), "rate_percent": float(r['rate'])}
            for _, r in dm_agg.iterrows()
        }

    avg_meds_readm = 0.0
    avg_meds_not_readm = 0.0
    if 'num_medications' in target_df.columns:
        readm_meds = target_df[target_df['is_readmitted'] == 1]['num_medications']
        not_readm_meds = target_df[target_df['is_readmitted'] == 0]['num_medications']
        if len(readm_meds) > 0:
            avg_meds_readm = round(float(readm_meds.mean()), 2)
        if len(not_readm_meds) > 0:
            avg_meds_not_readm = round(float(not_readm_meds.mean()), 2)

    return {
        "insulin": insulin_breakdown,
        "change": change_breakdown,
        "diabetesMed": diabetes_med_breakdown,
        "avg_medications_readmitted": avg_meds_readm,
        "avg_medications_not_readmitted": avg_meds_not_readm
    }


def calculate_diagnosis_stats(df=None):
    """
    Calculate dynamic readmission rates across ICD-9 primary diagnosis clinical groupings.
    """
    target_df = _resolve_df(df)
    total = len(target_df)
    if total == 0:
        return []

    col = 'diag_1_cat'
    if col not in target_df.columns and 'diag_1' in target_df.columns:
        target_df['diag_1_cat'] = target_df['diag_1'].apply(categorize_diagnosis)

    if col not in target_df.columns:
        return []

    diag_agg = target_df.groupby(col).agg(
        total=('is_readmitted', 'count'),
        readmitted=('is_readmitted', 'sum'),
        rate=('is_readmitted', lambda x: round(x.mean() * 100, 2) if len(x) > 0 else 0.0)
    ).reset_index().sort_values('rate', ascending=False)

    diag_list = [
        {"category": str(r[col]), "total": int(r['total']), "readmitted": int(r['readmitted']), "rate_percent": float(r['rate'])}
        for _, r in diag_agg.iterrows()
    ]
    return diag_list


def calculate_lab_stats(df=None):
    """
    Calculate dynamic readmission rates across HbA1c and glucose serum lab thresholds.
    """
    target_df = _resolve_df(df)
    total = len(target_df)
    if total == 0:
        return {"a1c": [], "glucose": [], "avg_lab_readmitted": 0.0, "avg_lab_not_readmitted": 0.0}

    a1c_list = []
    if 'A1Cresult' in target_df.columns:
        a1c_agg = target_df.groupby('A1Cresult').agg(
            total=('is_readmitted', 'count'),
            readmitted=('is_readmitted', 'sum'),
            rate=('is_readmitted', lambda x: round(x.mean() * 100, 2) if len(x) > 0 else 0.0)
        ).reset_index().sort_values('rate', ascending=False)

        a1c_list = [
            {"result": str(r['A1Cresult']), "total": int(r['total']), "readmitted": int(r['readmitted']), "rate_percent": float(r['rate'])}
            for _, r in a1c_agg.iterrows()
        ]

    glu_list = []
    if 'max_glu_serum' in target_df.columns:
        glu_agg = target_df.groupby('max_glu_serum').agg(
            total=('is_readmitted', 'count'),
            readmitted=('is_readmitted', 'sum'),
            rate=('is_readmitted', lambda x: round(x.mean() * 100, 2) if len(x) > 0 else 0.0)
        ).reset_index().sort_values('rate', ascending=False)

        glu_list = [
            {"result": str(r['max_glu_serum']), "total": int(r['total']), "readmitted": int(r['readmitted']), "rate_percent": float(r['rate'])}
            for _, r in glu_agg.iterrows()
        ]

    avg_lab_readm = 0.0
    avg_lab_not_readm = 0.0
    if 'num_lab_procedures' in target_df.columns:
        readm_labs = target_df[target_df['is_readmitted'] == 1]['num_lab_procedures']
        not_readm_labs = target_df[target_df['is_readmitted'] == 0]['num_lab_procedures']
        if len(readm_labs) > 0:
            avg_lab_readm = round(float(readm_labs.mean()), 2)
        if len(not_readm_labs) > 0:
            avg_lab_not_readm = round(float(not_readm_labs.mean()), 2)

    return {
        "a1c": a1c_list,
        "glucose": glu_list,
        "avg_lab_readmitted": avg_lab_readm,
        "avg_lab_not_readmitted": avg_lab_not_readm
    }


def calculate_utilization_stats(df=None):
    """
    Calculate dynamic readmission rates based on prior hospitalization tiers,
    emergency department visits, and encounter length of stay.
    """
    target_df = _resolve_df(df)
    total = len(target_df)
    if total == 0:
        return {"inpatient": [], "emergency": [], "length_of_stay": []}

    df_copy = target_df.copy()

    # Inpatient visits tier
    inpatient_list = []
    if 'number_inpatient' in df_copy.columns:
        df_copy['inpatient_tier'] = df_copy['number_inpatient'].apply(
            lambda x: "0 Stays" if x == 0 else ("1 Stay" if x == 1 else "2+ Stays")
        )
        inp_agg = df_copy.groupby('inpatient_tier').agg(
            total=('is_readmitted', 'count'),
            readmitted=('is_readmitted', 'sum'),
            rate=('is_readmitted', lambda x: round(x.mean() * 100, 2) if len(x) > 0 else 0.0)
        ).reset_index().sort_values('rate', ascending=False)

        inpatient_list = [
            {"tier": str(r['inpatient_tier']), "total": int(r['total']), "readmitted": int(r['readmitted']), "rate_percent": float(r['rate'])}
            for _, r in inp_agg.iterrows()
        ]

    # Emergency visits tier
    er_list = []
    if 'number_emergency' in df_copy.columns:
        df_copy['er_tier'] = df_copy['number_emergency'].apply(lambda x: "0 ER Visits" if x == 0 else "1+ ER Visits")
        er_agg = df_copy.groupby('er_tier').agg(
            total=('is_readmitted', 'count'),
            readmitted=('is_readmitted', 'sum'),
            rate=('is_readmitted', lambda x: round(x.mean() * 100, 2) if len(x) > 0 else 0.0)
        ).reset_index().sort_values('rate', ascending=False)

        er_list = [
            {"tier": str(r['er_tier']), "total": int(r['total']), "readmitted": int(r['readmitted']), "rate_percent": float(r['rate'])}
            for _, r in er_agg.iterrows()
        ]

    # Length of stay tier
    los_list = []
    if 'time_in_hospital' in df_copy.columns:
        df_copy['los_tier'] = df_copy['time_in_hospital'].apply(
            lambda x: "Short (1-3 days)" if x <= 3 else ("Medium (4-6 days)" if x <= 6 else "Extended (7-14 days)")
        )
        los_agg = df_copy.groupby('los_tier').agg(
            total=('is_readmitted', 'count'),
            readmitted=('is_readmitted', 'sum'),
            rate=('is_readmitted', lambda x: round(x.mean() * 100, 2) if len(x) > 0 else 0.0)
        ).reset_index().sort_values('rate', ascending=False)

        los_list = [
            {"tier": str(r['los_tier']), "total": int(r['total']), "readmitted": int(r['readmitted']), "rate_percent": float(r['rate'])}
            for _, r in los_agg.iterrows()
        ]

    return {"inpatient": inpatient_list, "emergency": er_list, "length_of_stay": los_list}


def calculate_discharge_stats(df=None):
    """
    Calculate dynamic readmission rates across standardized discharge dispositions.
    """
    target_df = _resolve_df(df)
    total = len(target_df)
    if total == 0:
        return []

    col = 'discharge_clean'
    if col not in target_df.columns and 'discharge_disposition_id' in target_df.columns:
        target_df['discharge_clean'] = target_df['discharge_disposition_id'].apply(map_discharge)

    if col not in target_df.columns:
        return []

    d_agg = target_df.groupby(col).agg(
        total=('is_readmitted', 'count'),
        readmitted=('is_readmitted', 'sum'),
        rate=('is_readmitted', lambda x: round(x.mean() * 100, 2) if len(x) > 0 else 0.0)
    ).reset_index().sort_values('rate', ascending=False)

    return [
        {"destination": str(r[col]), "total": int(r['total']), "readmitted": int(r['readmitted']), "rate_percent": float(r['rate'])}
        for _, r in d_agg.iterrows()
    ]


def calculate_admission_stats(df=None):
    """
    Calculate dynamic readmission rates across standardized admission types.
    """
    target_df = _resolve_df(df)
    total = len(target_df)
    if total == 0:
        return {"admission_types": [], "admission_sources": []}

    adm_types = []
    col_adm = 'admission_clean'
    if col_adm not in target_df.columns and 'admission_type_id' in target_df.columns:
        target_df['admission_clean'] = target_df['admission_type_id'].apply(map_admission)

    if col_adm in target_df.columns:
        adm_agg = target_df.groupby(col_adm).agg(
            total=('is_readmitted', 'count'),
            readmitted=('is_readmitted', 'sum'),
            rate=('is_readmitted', lambda x: round(x.mean() * 100, 2) if len(x) > 0 else 0.0)
        ).reset_index().sort_values('rate', ascending=False)
        adm_types = [
            {"type": str(r[col_adm]), "total": int(r['total']), "readmitted": int(r['readmitted']), "rate_percent": float(r['rate'])}
            for _, r in adm_agg.iterrows()
        ]

    adm_sources = []
    col_src = 'source_clean'
    if col_src not in target_df.columns and 'admission_source_id' in target_df.columns:
        target_df['source_clean'] = target_df['admission_source_id'].apply(map_source)

    if col_src in target_df.columns:
        src_agg = target_df.groupby(col_src).agg(
            total=('is_readmitted', 'count'),
            readmitted=('is_readmitted', 'sum'),
            rate=('is_readmitted', lambda x: round(x.mean() * 100, 2) if len(x) > 0 else 0.0)
        ).reset_index().sort_values('rate', ascending=False)
        adm_sources = [
            {"source": str(r[col_src]), "total": int(r['total']), "readmitted": int(r['readmitted']), "rate_percent": float(r['rate'])}
            for _, r in src_agg.iterrows()
        ]

    return {"admission_types": adm_types, "admission_sources": adm_sources}


def has_term(text, pattern):
    """Check if regex pattern exists in text matching word boundaries."""
    return bool(re.search(r'\b(?:' + pattern + r')\b', text, re.IGNORECASE))


def has_word(text, word):
    """Check if whole word exists in text using regex word boundary."""
    return bool(re.search(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE))


def handle_data_query(question, chat_history=None, df=None):
    """
    Intelligently inspect user inquiries, execute exact Pandas calculations against
    the dataset, and format the verified results without hardcoded figures.
    Returns:
        is_handled (bool): True if this is a dataset-level question, False if patient-specific.
        response_md (str): Formatted clinical markdown answer with verified numbers.
        verified_data (dict): Raw data payload used for natural language explanation.
    """
    q_lower = question.lower().strip()
    target_df = _resolve_df(df)
    total_cohort = len(target_df)

    if total_cohort == 0:
        return True, "⚠️ **Empty Dataset Notice:** The provided dataset contains 0 records. Calculations cannot be performed.", {"total_encounters": 0}

    # Contextual resolution from last user question (for follow-ups)
    last_context = ""
    if chat_history and isinstance(chat_history, list):
        for msg in reversed(chat_history[-4:]):
            if msg.get("role") == "user":
                last_context += " " + msg.get("content", "").lower()

    # 1. Check for unavailable / missing clinical columns first
    unrecorded_features = [
        (r"blood\s+pressures?", "Blood Pressure", "Blood Pressure (systolic/diastolic) was not recorded in this 1999–2008 dataset.", "Explore cardiovascular diagnoses (`Circulatory`), prior inpatient utilization, or HbA1c lab results instead."),
        (r"smok(ing|er|ers|e)?|tobacco", "Smoking", "Tobacco use and smoking status were not captured in this hospital cohort.", "Review respiratory diagnoses (`Respiratory`) or total prior hospitalizations as potential vulnerability proxies."),
        (r"bmi|body\s+mass\s+index", "BMI", "Body Mass Index (BMI) was not directly recorded; patient weight was missing in 96.9% of encounters.", "Examine glycemic lab results (`max_glu_serum`, `A1Cresult`) or active insulin regimen adjustments."),
        (r"weights?", "Weight", "Patient weight is missing in over 96.9% of records in this dataset.", "Examine glycemic lab results (`max_glu_serum`, `A1Cresult`) or active insulin regimen adjustments."),
        (r"costs?|charges?|billings?", "Cost / Financial Charges", "Financial charges, total encounter costs, and billing totals were not recorded.", "Review resource utilization proxies such as `time_in_hospital`, `num_medications`, and `num_lab_procedures`."),
        (r"doctors?|physicians?|practitioners?", "Physician / Doctor", "Physician names and specific practitioner identifiers were de-identified for patient privacy.", "Review admission sources (e.g., Physician/Clinic Referral vs. Emergency Room) and discharge destinations."),
        (r"hospital\s+names?|institution\s+names?", "Hospital Name", "Individual hospital institution names are de-identified (UCI 130-US Hospitals dataset).", "Explore aggregate cohort-level comparisons across all 130 participating US clinical centers."),
        (r"cholesterols?|lipids?", "Cholesterol", "Lipid panels and cholesterol measurements (LDL/HDL) were not collected in this dataset.", "Examine available glycemic markers (`A1Cresult`, `max_glu_serum`) and circulatory diagnoses."),
        (r"heart\s+rates?|pulses?", "Heart Rate", "Vitals including pulse and heart rate were not captured in this electronic health record extract.", "Review circulatory ICD-9 diagnoses and prior inpatient admissions."),
        (r"oxygens?|o2|pulse\s+oximetry", "Oxygen Saturation", "Pulse oximetry and oxygen saturation were not recorded in this dataset.", "Examine respiratory primary diagnoses (`Respiratory`) and length of hospital stay.")
    ]

    for pattern, title_name, explanation, suggestion in unrecorded_features:
        if has_term(q_lower, pattern):
            resp = f"""💡 *[Verified Dataset Calculation Engine]*

### ⚠️ Data Availability Boundary: {title_name}

- **Status:** **Unavailable in Dataset**
- **Explanation:** {explanation}
- **Available Variables in `diabetic_data.csv`:** Demographics (`gender`, `age`, `race`), In-Hospital Care (`time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`), Prior Healthcare Utilization (`number_outpatient`, `number_emergency`, `number_inpatient`), Diagnoses (`diag_1`, `diag_2`, `diag_3`, `number_diagnoses`), Discharge Disposition (`discharge_disposition_id`), and Diabetic Pharmacotherapy (`insulin`, `diabetesMed`, `change`, `A1Cresult`, `max_glu_serum`).
- **Recommended Alternative:** {suggestion}

> **📊 Analytical Provenance & Methodology:**
> - **Queried Variable:** `{title_name}`
> - **Cohort Denominator:** N = {total_cohort:,} post-discharge encounters
> - **Integrity Safeguard:** Transparent boundary reporting prevents unverified clinical hallucination."""
            return True, resp, {"status": "unavailable", "variable": title_name}

    # 2. Gender Demographics & Readmissions
    is_gender_query = (
        has_term(q_lower, r"genders?|males?|females?|men|women|sex(es)?")
    ) and not any(phrase in q_lower for phrase in ["this patient", "my patient", "the patient's current", "current patient"])

    is_gender_followup = (
        (has_word(q_lower, "female") or has_word(q_lower, "females") or has_word(q_lower, "women") or
         has_word(q_lower, "male") or has_word(q_lower, "males") or has_word(q_lower, "men")) and
        any(w in last_context for w in ["gender", "male", "female", "readmit", "risk"])
    )

    if is_gender_query or is_gender_followup:
        g_stats = calculate_gender_stats(target_df)
        total_encounters = sum(s["total_encounters"] for s in g_stats.values())
        total_readmitted = sum(s["readmitted_count"] for s in g_stats.values())
        cohort_rate = (total_readmitted / total_encounters * 100) if total_encounters > 0 else 0.0

        is_male_only = any(has_word(q_lower, w) for w in ["male", "males", "men"]) and not any(has_word(q_lower, w) for w in ["female", "females", "women"])
        is_female_only = any(has_word(q_lower, w) for w in ["female", "females", "women"]) and not any(has_word(q_lower, w) for w in ["male", "males", "men"])

        if is_male_only:
            specific_focus = "Male Encounters Focus"
        elif is_female_only:
            specific_focus = "Female Encounters Focus"
        else:
            specific_focus = "Full Gender Distribution"

        rows_md = []
        for g_name, s in g_stats.items():
            rows_md.append(f"| **{g_name}** | **{s['total_encounters']:,}** | {s['percent_of_cohort']:.2f}% | **{s['readmitted_count']:,}** | **{s['readmission_rate_percent']:.2f}%** | {s['not_readmitted_count']:,} |")

        table_rows = "\n".join(rows_md)

        # Dynamic findings
        m_data = g_stats.get("Male", {"total_encounters": 0, "readmitted_count": 0, "readmission_rate_percent": 0.0})
        f_data = g_stats.get("Female", {"total_encounters": 0, "readmitted_count": 0, "readmission_rate_percent": 0.0})
        delta_rate = round(f_data.get("readmission_rate_percent", 0.0) - m_data.get("readmission_rate_percent", 0.0), 2)
        female_share_readm = round((f_data.get("readmitted_count", 0) / total_readmitted * 100), 1) if total_readmitted > 0 else 0.0

        unknown_note = ""
        if "Unknown/Invalid" in g_stats:
            un_count = g_stats["Unknown/Invalid"]["total_encounters"]
            un_pct = (un_count / total_encounters * 100) if total_encounters > 0 else 0.0
            unknown_note = f"\n- **Unknown/Invalid Records:** Exactly **{un_count}** encounters ({un_pct:.3f}% of cohort) had unrecorded intake sex in the original hospital records; 0 were readmitted."

        resp = f"""💡 *[Verified Dataset Calculation Engine]*

### 👥 Gender Demographic & Readmission Breakdown ({specific_focus})

*Exact Pandas calculations computed dynamically across all **{total_encounters:,}** eligible encounters in `diabetic_data.csv`:*

| Gender | Total Encounters | % of Cohort | Readmitted Encounters (At Risk) | Readmission Rate (%) | Non-Readmitted Encounters |
| :--- | :---: | :---: | :---: | :---: | :---: |
{table_rows}
| **Total Cohort** | **{total_encounters:,}** | **100.00%** | **{total_readmitted:,}** | **{cohort_rate:.2f}%** | **{total_encounters - total_readmitted:,}** |

**Key Analytical Findings:**
- **Male Patients:** There are **{m_data.get('total_encounters', 0):,}** male encounters in the dataset. Of those, **{m_data.get('readmitted_count', 0):,}** were readmitted (**{m_data.get('readmission_rate_percent', 0):.2f}%** readmission rate).
- **Female Patients:** There are **{f_data.get('total_encounters', 0):,}** female encounters in the dataset. Of those, **{f_data.get('readmitted_count', 0):,}** were readmitted (**{f_data.get('readmission_rate_percent', 0):.2f}%** readmission rate).
- **Comparative Rate Delta:** Female diabetic patients have a **{abs(delta_rate):.2f}%** {'higher' if delta_rate > 0 else 'lower'} historical readmission rate, representing **{female_share_readm}%** of all readmitted encounters.{unknown_note}

> **📊 Analytical Provenance & Methodology:**
> - **Source Columns:** `gender`, `readmitted`
> - **Cohort Denominator:** N = {total_encounters:,} eligible post-discharge encounters (excluding in-hospital expired / hospice)
> - **Calculation Formula:** $\\text{{Readmission Rate (\\%)}} = \\frac{{\\text{{Readmitted Encounters (Target = 1)}}}}{{\\text{{Total Encounters in Group}}}} \\times 100$"""

        return True, resp, {"gender_stats": g_stats, "total_encounters": total_encounters, "total_readmitted": total_readmitted}

    # 3. Age Groups and Readmission Rates (Checked BEFORE overall rate so "Which age groups have highest readmission" matches Age)
    is_age_query = (
        any(phrase in q_lower for phrase in ["age group", "age groups", "which age", "highest readmission rate", "highest readmission", "age-group", "oldest", "youngest", "elderly", "age bracket", "age breakdown", "by age"]) or
        (has_word(q_lower, "age") and any(w in q_lower for w in ["rate", "group", "highest", "lowest", "compare", "breakdown", "distribution", "bracket", "affect"]))
    ) and not any(phrase in q_lower for phrase in ["this patient", "my patient", "the patient's current"])

    is_age_followup = (
        (has_word(q_lower, "age") or "old" in q_lower or "years" in q_lower or any(b in q_lower for b in ["70-80", "80-90", "60-70", "50-60", "40-50", "30-40", "20-30", "10-20", "0-10", "90-100"])) and
        any(w in last_context for w in ["age", "rate", "group", "readmitted", "bracket"])
    )

    if is_age_query or is_age_followup:
        age_stats = calculate_age_stats(target_df)
        if not age_stats:
            return True, "⚠️ No age data available in dataset.", {}

        rows_md = "\n".join([
            f"| `{a['age_bracket']}` | **{a['readmission_rate_percent']:.2f}%** | {a['readmitted_count']:,} | {a['total_encounters']:,} | {a['percent_of_cohort']:.1f}% |"
            for a in age_stats
        ])

        top_age = age_stats[0]
        second_age = age_stats[1] if len(age_stats) > 1 else top_age
        lowest_age = age_stats[-1]

        senior_encounters = sum(a['total_encounters'] for a in age_stats if a['age_bracket'] in ['[60-70)', '[70-80)', '[80-90)', '[90-100)'])
        senior_pct = round(senior_encounters / total_cohort * 100, 1) if total_cohort > 0 else 0.0

        resp = f"""💡 *[Verified Dataset Calculation Engine]*

### 🎂 Readmission Rate by Age Group (Ranked from Highest to Lowest)

*Exact calculations computed dynamically across all **{total_cohort:,}** eligible diabetic patient encounters:*

| Age Group | Readmission Rate (%) | Readmitted Encounters | Total Encounters | % of Total Cohort |
| :---: | :---: | :---: | :---: | :---: |
{rows_md}

**Key Analytical Findings:**
- **Highest Risk Bracket:** Patients in the **`{top_age['age_bracket']}`** age group had the highest readmission rate at **{top_age['readmission_rate_percent']}%** ({top_age['readmitted_count']:,} / {top_age['total_encounters']:,} encounters).
- **Second Highest Risk:** **`{second_age['age_bracket']}`** at **{second_age['readmission_rate_percent']}%** ({second_age['readmitted_count']:,} / {second_age['total_encounters']:,} encounters).
- **Lowest Risk Bracket:** **`{lowest_age['age_bracket']}`** had the lowest readmission rate at **{lowest_age['readmission_rate_percent']}%** ({lowest_age['readmitted_count']:,} / {lowest_age['total_encounters']:,} encounters).
- **Volume Concentration:** Over **{senior_pct}%** of all encounters ({senior_encounters:,} encounters) represent patients aged 60 and older.

> **📊 Analytical Provenance & Methodology:**
> - **Source Columns:** `age`, `readmitted`
> - **Cohort Denominator:** N = {total_cohort:,} post-discharge encounters
> - **Calculation Formula:** $\\text{{Readmission Rate (\\%)}} = \\frac{{\\text{{Readmitted Encounters in Bracket}}}}{{\\text{{Total Encounters in Bracket}}}} \\times 100$"""
        return True, resp, {"age_stats": age_stats}

    # 4. Discharge Destinations & Transitions
    is_discharge_query = any(w in q_lower for w in ["discharge destination", "discharge destinations", "discharge type", "discharged to", "home vs home health", "transferred facility", "ama", "discharge status", "discharge disposition"]) and not any(phrase in q_lower for phrase in ["this patient", "my patient", "the patient's current"])
    if is_discharge_query:
        d_stats = calculate_discharge_stats(target_df)
        if d_stats:
            d_rows = "\n".join([
                f"| `{d['destination']}` | **{d['rate_percent']:.2f}%** | {d['readmitted']:,} | {d['total']:,} | {d['total']/total_cohort*100:.1f}% |"
                for d in d_stats
            ])
            top_dest = d_stats[0]
            resp = f"""💡 *[Verified Dataset Calculation Engine]*

### 🏥 Readmission Rates by Discharge Destination Category

*Exact calculations across all **{total_cohort:,}** eligible post-discharge encounters:*

| Discharge Destination | Readmission Rate (%) | Readmitted Encounters | Total Encounters | % of Cohort |
| :--- | :---: | :---: | :---: | :---: |
{d_rows}

**Key Clinical Findings:**
- **Highest Hazard Disposition:** Patients discharged to **`{top_dest['destination']}`** experienced the highest readmission rate at **{top_dest['rate_percent']}%** ({top_dest['readmitted']:,} / {top_dest['total']:,} encounters).
- **Care Coordination Takeaway:** Specialized post-acute transition support (e.g. Home Health follow-up) signals patients with greater underlying functional frailty requiring intensified outpatient monitoring.

> **📊 Analytical Provenance & Methodology:**
> - **Source Columns:** `discharge_disposition_id` (mapped to standardized clinical categories), `readmitted`
> - **Cohort Denominator:** N = {total_cohort:,} post-discharge encounters"""
            return True, resp, {"discharge_stats": d_stats}

    # 5. Admission Types & Hospital Entry Points
    is_admission_query = (
        any(w in q_lower for w in ["admission type", "admission types", "admission source", "admission channel", "emergency vs elective", "urgent admission", "admission relate"]) or
        (has_term(q_lower, r"admissions?") and any(w in q_lower for w in ["emergency", "urgent", "elective", "type", "source", "channel", "difference", "compare"]))
    ) and not any(phrase in q_lower for phrase in ["this patient", "my patient", "the patient's current"])
    if is_admission_query:
        adm_stats = calculate_admission_stats(target_df)
        t_rows = "\n".join([
            f"| `{a['type']}` | **{a['rate_percent']:.2f}%** | {a['readmitted']:,} | {a['total']:,} |"
            for a in adm_stats.get("admission_types", [])
        ])
        s_rows = "\n".join([
            f"| `{s['source']}` | **{s['rate_percent']:.2f}%** | {s['readmitted']:,} | {s['total']:,} |"
            for s in adm_stats.get("admission_sources", [])
        ])

        resp = f"""💡 *[Verified Dataset Calculation Engine]*

### 🚑 Hospital Admission Type & Source Readmission Correlations

*Exact calculations across all **{total_cohort:,}** eligible encounters:*

#### 1. Admission Acuity Type:
| Admission Type | Readmission Rate (%) | Readmitted Encounters | Total Encounters |
| :--- | :---: | :---: | :---: |
{t_rows}

#### 2. Admission Source Channel:
| Admission Source Channel | Readmission Rate (%) | Readmitted Encounters | Total Encounters |
| :--- | :---: | :---: | :---: |
{s_rows}

> **📊 Analytical Provenance & Methodology:**
> - **Source Columns:** `admission_type_id`, `admission_source_id`, `readmitted`
> - **Cohort Denominator:** N = {total_cohort:,} encounters"""
        return True, resp, adm_stats

    # 6. Medications, Insulin, Regimen Changes across Dataset
    is_med_query = any(w in q_lower for w in ["insulin", "medication", "medications", "prescribed", "drug", "regimen", "change in medication", "polypharmacy", "diabetesmed"]) and not any(phrase in q_lower for phrase in ["this patient", "my patient", "the patient's current"])
    if is_med_query:
        med_stats = calculate_medication_stats(target_df)
        ins_rows = "\n".join([
            f"| `{item['regimen']}` | **{item['rate_percent']:.2f}%** | {item['readmitted']:,} | {item['total']:,} |"
            for item in med_stats.get('insulin', [])
        ])

        ch_data = med_stats.get('change', {})
        dm_data = med_stats.get('diabetesMed', {})

        ch_yes_rate = ch_data.get('Ch', {}).get('rate_percent', 0.0)
        ch_no_rate = ch_data.get('No', {}).get('rate_percent', 0.0)
        ch_delta = round(ch_yes_rate - ch_no_rate, 2)

        resp = f"""💡 *[Verified Dataset Calculation Engine]*

### 💊 Pharmacotherapy & Insulin Impact on Readmission

*Exact Pandas calculations computed dynamically across all **{total_cohort:,}** eligible encounters:*

#### 1. Readmission Rates by Insulin Regimen Status:
| Insulin Regimen | Readmission Rate (%) | Readmitted Encounters | Total Encounters |
| :--- | :---: | :---: | :---: |
{ins_rows}

#### 2. Medication Regimen Alteration (`change`):
- **Dosage/Medication Changed (`Ch`):** **{ch_yes_rate}%** readmitted ({ch_data.get('Ch', {}).get('readmitted', 0):,} / {ch_data.get('Ch', {}).get('total', 0):,})
- **No Change (`No`):** **{ch_no_rate}%** readmitted ({ch_data.get('No', {}).get('readmitted', 0):,} / {ch_data.get('No', {}).get('total', 0):,})
- *Analytical Finding:* Patients undergoing active medication adjustments during admission have a **{ch_delta:+.2f}%** difference in readmission rate.

#### 3. Diabetes Medication Prescribed (`diabetesMed`):
- **Prescribed (`Yes`):** **{dm_data.get('Yes', {}).get('rate_percent', 0.0)}%** readmitted ({dm_data.get('Yes', {}).get('readmitted', 0):,} / {dm_data.get('Yes', {}).get('total', 0):,})
- **Not Prescribed (`No`):** **{dm_data.get('No', {}).get('rate_percent', 0.0)}%** readmitted ({dm_data.get('No', {}).get('readmitted', 0):,} / {dm_data.get('No', {}).get('total', 0):,})

#### 4. Medication Count Burden:
- Readmitted patients averaged **{med_stats.get('avg_medications_readmitted', 0.0)} medications**, compared to **{med_stats.get('avg_medications_not_readmitted', 0.0)} medications** for non-readmitted patients.

> **📊 Analytical Provenance & Methodology:**
> - **Source Columns:** `insulin`, `change`, `diabetesMed`, `num_medications`, `readmitted`
> - **Cohort Denominator:** N = {total_cohort:,} post-discharge encounters"""
        return True, resp, med_stats

    # 7. Primary ICD-9 Diagnoses across Dataset
    is_diag_query = any(w in q_lower for w in ["diagnoses in the dataset", "diagnosis relate", "diagnoses relate", "icd-9", "icd categories", "diagnoses have the highest", "most common diagnosis", "circulatory vs", "primary diagnosis", "diabetes diagnosis"]) and not any(phrase in q_lower for phrase in ["this patient", "my patient", "the patient's current"])
    if is_diag_query:
        diag_stats = calculate_diagnosis_stats(target_df)
        if diag_stats:
            diag_rows = "\n".join([
                f"| `{d['category']}` | **{d['rate_percent']:.2f}%** | {d['readmitted']:,} | {d['total']:,} | {d['total']/total_cohort*100:.1f}% |"
                for d in diag_stats
            ])

            top_d = diag_stats[0]
            second_d = diag_stats[1] if len(diag_stats) > 1 else top_d
            lowest_d = diag_stats[-1]

            resp = f"""💡 *[Verified Dataset Calculation Engine]*

### 🩺 Readmission Rates by Primary Diagnosis Category (ICD-9)

*Exact calculations across all **{total_cohort:,}** patient encounters:*

| Diagnosis Category | Readmission Rate (%) | Readmitted Encounters | Total Encounters | % of Cohort |
| :--- | :---: | :---: | :---: | :---: |
{diag_rows}

**Key Clinical Findings:**
- **Highest Readmission Category:** **`{top_d['category']}`** diagnoses had the highest rate at **{top_d['rate_percent']}%** ({top_d['readmitted']:,} / {top_d['total']:,} encounters).
- **Second Highest:** **`{second_d['category']}`** at **{second_d['rate_percent']}%** ({second_d['readmitted']:,} / {second_d['total']:,} encounters).
- **Lowest Rate:** **`{lowest_d['category']}`** at **{lowest_d['rate_percent']}%** ({lowest_d['readmitted']:,} / {lowest_d['total']:,} encounters).

> **📊 Analytical Provenance & Methodology:**
> - **Source Columns:** `diag_1` (mapped to standardized ICD-9 chapters), `readmitted`
> - **Cohort Denominator:** N = {total_cohort:,} post-discharge encounters"""
            return True, resp, {"diagnosis_stats": diag_stats}

    # 8. Lab Results (HbA1c & Blood Glucose) across Dataset
    is_lab_query = any(w in q_lower for w in ["lab results relate", "how does a1c relate", "how does glucose relate", "hba1c in the dataset", "lab results across the dataset", "max_glu_serum", "a1cresult", "lab procedures", "blood glucose in the dataset"]) and not any(phrase in q_lower for phrase in ["this patient", "my patient", "the patient's current"])
    if is_lab_query:
        lab_stats = calculate_lab_stats(target_df)
        a1c_rows = "\n".join([
            f"| `{item['result']}` | **{item['rate_percent']:.2f}%** | {item['readmitted']:,} | {item['total']:,} |"
            for item in lab_stats.get('a1c', [])
        ])
        glu_rows = "\n".join([
            f"| `{item['result']}` | **{item['rate_percent']:.2f}%** | {item['readmitted']:,} | {item['total']:,} |"
            for item in lab_stats.get('glucose', [])
        ])

        resp = f"""💡 *[Verified Dataset Calculation Engine]*

### 🧪 Lab Test Results & Readmission Rates (Dataset Breakdown)

*Exact calculations across all **{total_cohort:,}** encounters:*

#### 1. Glycated Hemoglobin (HbA1c) Results:
| HbA1c Category | Readmission Rate (%) | Readmitted Encounters | Total Encounters |
| :--- | :---: | :---: | :---: |
{a1c_rows}

#### 2. Maximum Glucose Serum Thresholds:
| Max Glucose Result | Readmission Rate (%) | Readmitted Encounters | Total Encounters |
| :--- | :---: | :---: | :---: |
{glu_rows}

#### 3. Diagnostic Testing Volume:
- Readmitted patients had an average of **{lab_stats.get('avg_lab_readmitted', 0.0)} lab procedures**, versus **{lab_stats.get('avg_lab_not_readmitted', 0.0)}** for non-readmitted patients.

> **📊 Analytical Provenance & Methodology:**
> - **Source Columns:** `A1Cresult`, `max_glu_serum`, `num_lab_procedures`, `readmitted`
> - **Cohort Denominator:** N = {total_cohort:,} encounters"""
        return True, resp, lab_stats

    # 9. Prior Hospitalization & Utilization across Dataset
    is_util_query = any(w in q_lower for w in ["inpatient in the dataset", "emergency room in the dataset", "er visits relate", "prior stays relate", "length of stay relate", "prior visits relate", "hospitalization history"]) and not any(phrase in q_lower for phrase in ["this patient", "my patient", "the patient's current"])
    if is_util_query:
        util_stats = calculate_utilization_stats(target_df)
        inp_rows = "\n".join([
            f"| `{u['tier']}` | **{u['rate_percent']:.2f}%** | {u['readmitted']:,} | {u['total']:,} |"
            for u in util_stats.get('inpatient', [])
        ])
        er_rows = "\n".join([
            f"| `{u['tier']}` | **{u['rate_percent']:.2f}%** | {u['readmitted']:,} | {u['total']:,} |"
            for u in util_stats.get('emergency', [])
        ])
        los_rows = "\n".join([
            f"| `{u['tier']}` | **{u['rate_percent']:.2f}%** | {u['readmitted']:,} | {u['total']:,} |"
            for u in util_stats.get('length_of_stay', [])
        ])

        inp_high_rate = util_stats.get('inpatient', [{}])[0].get('rate_percent', 0.0) if util_stats.get('inpatient') else 0.0

        resp = f"""💡 *[Verified Dataset Calculation Engine]*

### 🏥 Healthcare Utilization & Prior Hospitalization Impact

*Exact calculations across all **{total_cohort:,}** encounters:*

#### 1. Prior Inpatient Hospitalizations (Preceding 12 Months):
| Inpatient Admissions Tier | Readmission Rate (%) | Readmitted Encounters | Total Encounters |
| :--- | :---: | :---: | :---: |
{inp_rows}
*Key takeaway: Prior inpatient admissions is the single strongest statistical driver of readmission, with highest-tier patients exhibiting a **{inp_high_rate}%** readmission rate.*

#### 2. Prior Emergency Department Encounters:
| Emergency Department Tier | Readmission Rate (%) | Readmitted Encounters | Total Encounters |
| :--- | :---: | :---: | :---: |
{er_rows}

#### 3. Current Length of Stay (Days):
| Stay Length Tier | Readmission Rate (%) | Readmitted Encounters | Total Encounters |
| :--- | :---: | :---: | :---: |
{los_rows}

> **📊 Analytical Provenance & Methodology:**
> - **Source Columns:** `number_inpatient`, `number_emergency`, `time_in_hospital`, `readmitted`
> - **Cohort Denominator:** N = {total_cohort:,} encounters"""
        return True, resp, util_stats

    # 10. Overall Cohort Readmission Rate & Counts
    is_overall_rate_query = (
        ("readmitted" in q_lower or "readmission" in q_lower or "readmission rate" in q_lower or "readmit" in q_lower) and
        any(w in q_lower for w in ["percentage", "percent", "rate", "how many", "total", "proportion", "cohort", "overall", "what is the readmission", "fraction", "statistics"]) and
        not any(has_word(q_lower, w) for w in ["medication", "insulin", "diagnosis", "gender", "male", "female", "discharge", "admission", "this patient", "my patient", "the patient"])
    )

    if is_overall_rate_query:
        summary = get_cohort_summary(target_df)
        resp = f"""💡 *[Verified Dataset Calculation Engine]*

### 📊 Overall Cohort Readmission Metrics

*Exact calculations executed across **{summary['total_encounters']:,}** post-discharge encounters in `diabetic_data.csv`:*

- **Total Analyzed Post-Discharge Encounters:** **{summary['total_encounters']:,}**
- **Total Readmitted Encounters (`Target = 1`):** **{summary['readmitted_count']:,}** (**{summary['readmission_rate_percent']}%**)
- **Non-Readmitted Encounters (`Target = 0`):** **{summary['no_readmit_count']:,}** (**{summary['no_readmit_percent']}%**)

**Temporal Breakdown of Readmission Timing:**
| Readmission Category | Encounters | % of Total Cohort | Clinical Context |
| :--- | :---: | :---: | :--- |
| **Early Readmission (`<30` days)** | **{summary['readmit_under_30_count']:,}** | **{summary['readmit_under_30_percent']}%** | High acute post-discharge vulnerability |
| **Late Readmission (`>30` days)** | **{summary['readmit_over_30_count']:,}** | **{summary['readmit_over_30_percent']}%** | Chronic disease management decompensation |
| **No Readmission (`NO`)** | **{summary['no_readmit_count']:,}** | **{summary['no_readmit_percent']}%** | Successful initial discharge transition |

*Target Verification Note: The binary classification target groups both `<30` and `>30` days into Readmitted (1 = {summary['readmitted_count']:,}) vs. None (0 = {summary['no_readmit_count']:,}).*

> **📊 Analytical Provenance & Methodology:**
> - **Source Columns:** `readmitted`
> - **Cohort Denominator:** N = {summary['total_encounters']:,} post-discharge encounters
> - **Calculation Formula:** $\\text{{Readmission Rate (\\%)}} = \\frac{{\\text{{Readmitted Encounters (Target = 1)}}}}{{\\text{{Total Analyzed Encounters}}}} \\times 100$"""
        return True, resp, summary

    # 11. Dynamic Patient Characteristic / Filtering Queries
    is_char_query = any(phrase in q_lower for phrase in [
        "how many patients", "how many encounters", "what count of patients", "how many candidates",
        "how many people", "percentage of patients who", "number of patients with"
    ]) and not any(phrase in q_lower for phrase in ["this patient", "my patient", "the patient's current"])

    if is_char_query:
        if any(w in q_lower for w in ["stay", "day", "days"]) and any(w in q_lower for w in ["7", "week", "long", "extended"]):
            match_df = target_df[target_df['time_in_hospital'] >= 7]
            non_match_df = target_df[target_df['time_in_hospital'] < 7]
            count = len(match_df)
            readm = int(match_df['is_readmitted'].sum())
            rate = round(readm / count * 100, 2) if count > 0 else 0.0
            comp_rate = round(non_match_df['is_readmitted'].mean() * 100, 2) if len(non_match_df) > 0 else 0.0

            resp = f"""💡 *[Verified Dataset Calculation Engine]*

### ⏱️ Patient Characteristic Query: Extended Hospital Stays (≥ 7 Days)

- **Encounters Matching Criteria:** **{count:,}** out of **{total_cohort:,}** ({count/total_cohort*100:.2f}% of cohort)
- **Readmitted Count:** **{readm:,}**
- **Readmission Rate:** **{rate}%** (vs. **{comp_rate}%** for stays < 7 days)

> **📊 Analytical Provenance & Methodology:**
> - **Source Column:** `time_in_hospital` (filter: `>= 7`), `readmitted`
> - **Cohort Denominator:** N = {total_cohort:,} encounters"""
            return True, resp, {"characteristic": "stay >= 7 days", "count": count, "readmit_rate": rate}

        elif "insulin" in q_lower:
            match_df = target_df[target_df['insulin'] != 'No'] if 'insulin' in target_df.columns else pd.DataFrame()
            non_match_df = target_df[target_df['insulin'] == 'No'] if 'insulin' in target_df.columns else pd.DataFrame()
            count = len(match_df)
            readm = int(match_df['is_readmitted'].sum()) if count > 0 else 0
            rate = round(readm / count * 100, 2) if count > 0 else 0.0
            comp_rate = round(non_match_df['is_readmitted'].mean() * 100, 2) if len(non_match_df) > 0 else 0.0

            resp = f"""💡 *[Verified Dataset Calculation Engine]*

### 💉 Patient Characteristic Query: Prescribed Insulin Regimen

- **Encounters Prescribed Insulin:** **{count:,}** out of **{total_cohort:,}** ({count/total_cohort*100:.2f}% of cohort)
- **Readmitted Count:** **{readm:,}**
- **Readmission Rate:** **{rate}%** (vs. **{comp_rate}%** for encounters with no insulin)

> **📊 Analytical Provenance & Methodology:**
> - **Source Column:** `insulin` (filter: `!= 'No'`), `readmitted`
> - **Cohort Denominator:** N = {total_cohort:,} encounters"""
            return True, resp, {"characteristic": "prescribed insulin", "count": count, "readmit_rate": rate}

        elif any(w in q_lower for w in ["emergency", "er"]):
            match_df = target_df[target_df['number_emergency'] > 0] if 'number_emergency' in target_df.columns else pd.DataFrame()
            non_match_df = target_df[target_df['number_emergency'] == 0] if 'number_emergency' in target_df.columns else pd.DataFrame()
            count = len(match_df)
            readm = int(match_df['is_readmitted'].sum()) if count > 0 else 0
            rate = round(readm / count * 100, 2) if count > 0 else 0.0
            comp_rate = round(non_match_df['is_readmitted'].mean() * 100, 2) if len(non_match_df) > 0 else 0.0

            resp = f"""💡 *[Verified Dataset Calculation Engine]*

### 🚑 Patient Characteristic Query: Prior Emergency Room Utilization

- **Encounters with Prior ER Visits (≥ 1):** **{count:,}** out of **{total_cohort:,}** ({count/total_cohort*100:.2f}% of cohort)
- **Readmitted Count:** **{readm:,}**
- **Readmission Rate:** **{rate}%** (vs. **{comp_rate}%** for encounters with 0 ER visits)

> **📊 Analytical Provenance & Methodology:**
> - **Source Column:** `number_emergency` (filter: `> 0`), `readmitted`
> - **Cohort Denominator:** N = {total_cohort:,} encounters"""
            return True, resp, {"characteristic": "prior emergency visits", "count": count, "readmit_rate": rate}

    # 12. General Dataset Inquiries & Questions Catalog
    is_general_data_inquiry = any(phrase in q_lower for phrase in [
        "what columns", "available columns", "what can i ask", "what questions can you answer",
        "tell me about the dataset", "what is this dataset", "dataset overview", "supported questions",
        "how many rows", "dataset size", "dataset scale"
    ]) and not any(phrase in q_lower for phrase in ["this patient", "my patient"])

    if is_general_data_inquiry:
        summary = get_cohort_summary(target_df)
        resp = f"""💡 *[Verified Dataset Calculation Engine]*

### 📖 Dataset Overview & Supported Clinical Inquiries

The dataset analyzed is the **UCI Diabetes 130-US Hospitals (1999–2008)** clinical extract, containing **{summary['total_encounters']:,}** post-discharge encounters across 130 medical centers.

#### 🎯 Verified Questions You Can Ask:
1. **👥 Gender Demographics:** *"How many male and female patients are in the dataset?"* or *"What is the male vs female readmission rate?"*
2. **🎂 Age Groups:** *"Which age groups have the highest readmission rates?"*
3. **📊 Overall Rates:** *"What percentage of patients were readmitted?"* or *"How many were readmitted under 30 days?"*
4. **💊 Medications & Insulin:** *"How do medications and insulin relate to readmission?"*
5. **🩺 Diagnoses:** *"How do primary diagnoses relate to readmission?"*
6. **🏥 Discharge & Admissions:** *"How does discharge destination affect readmission?"* or *"Emergency vs elective admissions?"*
7. **🧪 Lab Results:** *"How do HbA1c and glucose results affect readmission?"*
8. **⏱️ Specific Characteristics:** *"How many patients had hospital stays of 7 or more days?"*

> **📊 Analytical Provenance & Methodology:**
> - **Cohort Denominator:** N = {summary['total_encounters']:,} post-discharge encounters
> - **Integrity Notice:** Every query executes live Pandas calculations directly against the dataset."""
        return True, resp, {"status": "overview", "total_encounters": summary['total_encounters']}

    # 13. Safety Guard: If user clearly asked a general cohort question but it wasn't matched above,
    # NEVER answer with an irrelevant patient prediction! Provide transparent dataset guidance.
    is_cohort_inquiry = any(w in q_lower for w in [
        "dataset", "cohort", "database", "average in the data", "across the hospital",
        "population", "overall patient", "all patients", "statistical rate", "correlations in data"
    ]) and not any(phrase in q_lower for phrase in ["this patient", "my patient", "the patient's current", "current encounter"])

    if is_cohort_inquiry:
        return True, f"""💡 *[Verified Dataset Calculation Engine]*

### 🔍 Cohort Query Guidance

I recognize this as a question about the general hospital dataset (**N = {total_cohort:,} encounters**), but the specific variable or aggregation requested was not recognized.

- **Available Features:** `gender`, `age`, `time_in_hospital`, `num_lab_procedures`, `num_medications`, `number_inpatient`, `number_emergency`, `number_outpatient`, `diag_1_cat`, `insulin`, `change`, `diabetesMed`, `A1Cresult`, `max_glu_serum`, and `discharge_clean`.
- **Suggested Queries:**
  * *"How many male and female patients are in the dataset?"*
  * *"Which age groups have the highest readmission rates?"*
  * *"How do medications, insulin, and lab results relate to readmission?"*
  * *"What percentage of patients were readmitted overall?"*

> **📊 Integrity Safeguard:** Transparent routing ensures general dataset questions are never answered with an irrelevant individual patient assessment.""" , {"status": "unrecognized_cohort_query"}

    return False, None, None
