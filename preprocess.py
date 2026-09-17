import pandas as pd
import numpy as np

AGE_ORDER = [
    '[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
    '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'
]
AGE_MAP = {group: i for i, group in enumerate(AGE_ORDER)}

def categorize_diagnosis(val):
    """Categorize ICD-9 diagnosis codes into primary clinical groupings."""
    if pd.isna(val) or str(val).startswith('?') or str(val).strip() == 'Missing':
        return 'Other'
    val_str = str(val).strip()
    try:
        val_f = float(val_str)
        if (390 <= val_f <= 459) or val_f == 785:
            return 'Circulatory'
        elif (460 <= val_f <= 519) or val_f == 786:
            return 'Respiratory'
        elif (520 <= val_f <= 579) or val_f == 787:
            return 'Digestive'
        elif int(val_f) == 250:
            return 'Diabetes'
        elif 800 <= val_f <= 999:
            return 'Injury'
        elif 710 <= val_f <= 739:
            return 'Musculoskeletal'
        elif (580 <= val_f <= 629) or val_f == 788:
            return 'Genitourinary'
        elif 140 <= val_f <= 239:
            return 'Neoplasms'
        else:
            return 'Other'
    except ValueError:
        return 'Other'

def map_discharge(val):
    """Map raw discharge disposition IDs to standardized categories."""
    if val in [1]:
        return 'Home'
    elif val in [6, 8]:
        return 'Home Health'
    elif val in [2, 3, 4, 5, 22, 23, 24]:
        return 'Transferred Facility'
    elif val in [11, 13, 14, 19, 20, 21]:
        return 'Hospice/Expired'
    elif val == 7:
        return 'AMA'
    else:
        return 'Other'

def map_admission(val):
    """Map raw admission type IDs to standardized categories."""
    if val == 1:
        return 'Emergency'
    elif val == 2:
        return 'Urgent'
    elif val == 3:
        return 'Elective'
    else:
        return 'Other'

def map_source(val):
    """Map raw admission source IDs to standardized categories."""
    if val == 7:
        return 'Emergency Room'
    elif val in [1, 2]:
        return 'Physician/Clinic Referral'
    elif val in [4, 5, 6]:
        return 'Transfer'
    else:
        return 'Other'

def load_and_preprocess(filepath="diabetic_data.csv"):
    """
    Load and preprocess the hospital readmission dataset.
    Returns:
        X (pd.DataFrame): Training feature matrix
        y (pd.Series): Target vector (1 = readmitted, 0 = not readmitted)
        feature_columns (list): List of feature column names in order
        encoders (dict): Mapping dictionaries for categorical features
    """
    df = pd.read_csv(filepath)

    # Filter out expired patients (clinical standard: expired patients cannot be readmitted)
    df = df[~df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])].copy()

    # Binary target (1 if readmitted <30 or >30, 0 if not readmitted)
    y = df['readmitted'].apply(lambda x: 1 if x in ['<30', '>30'] else 0)

    # Feature transformations
    df['age_num'] = df['age'].map(lambda a: AGE_MAP.get(a, 5))
    df['diag_1_cat'] = df['diag_1'].apply(categorize_diagnosis)
    df['discharge_clean'] = df['discharge_disposition_id'].apply(map_discharge)
    df['admission_clean'] = df['admission_type_id'].apply(map_admission)
    df['source_clean'] = df['admission_source_id'].apply(map_source)

    features = [
        'age_num', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient',
        'number_diagnoses', 'diag_1_cat', 'discharge_clean', 'admission_clean', 'source_clean',
        'diabetesMed', 'change', 'insulin', 'A1Cresult', 'max_glu_serum'
    ]

    X = df[features].copy()

    # Engineered interaction features
    X['total_prior_visits'] = X['number_outpatient'] + X['number_emergency'] + X['number_inpatient']
    X['service_density'] = ((X['num_medications'] + X['num_lab_procedures']) / X['time_in_hospital']).round(2)
    X['prior_inpatient_ratio'] = (X['number_inpatient'] / (X['total_prior_visits'] + 1)).round(3)

    # Encode categoricals with explicit sorted dictionaries for consistent inference
    cat_cols = [
        'diag_1_cat', 'discharge_clean', 'admission_clean', 'source_clean',
        'diabetesMed', 'change', 'insulin', 'A1Cresult', 'max_glu_serum'
    ]
    encoders = {}
    for c in cat_cols:
        uniques = sorted(X[c].dropna().unique().tolist())
        mapping = {val: idx for idx, val in enumerate(uniques)}
        X[c] = X[c].map(lambda v: mapping.get(str(v), 0))
        encoders[c] = mapping

    feature_columns = X.columns.tolist()
    return X, y, feature_columns, encoders

def prepare_single_input(data_dict, feature_columns, encoders):
    """
    Transform a dictionary of user input into a model-ready 1-row DataFrame.
    """
    row = dict(data_dict)

    # Map age if provided as group string
    if isinstance(row.get('age'), str):
        row['age_num'] = AGE_MAP.get(row['age'], 5)
    elif 'age_num' not in row:
        row['age_num'] = int(row.get('age', 5))

    # Compute engineered features
    outpatient = int(row.get('number_outpatient', 0))
    emergency = int(row.get('number_emergency', 0))
    inpatient = int(row.get('number_inpatient', 0))
    total_prior = outpatient + emergency + inpatient
    row['total_prior_visits'] = total_prior

    time_hosp = max(1, int(row.get('time_in_hospital', 1)))
    meds = int(row.get('num_medications', 10))
    labs = int(row.get('num_lab_procedures', 40))
    row['service_density'] = round((meds + labs) / time_hosp, 2)
    row['prior_inpatient_ratio'] = round(inpatient / (total_prior + 1), 3)

    # Encode categoricals using saved mappings
    for col, mapping in encoders.items():
        val = str(row.get(col, ''))
        row[col] = mapping.get(val, 0)

    # Create DataFrame with exact column ordering
    df_row = pd.DataFrame([row])
    return df_row[feature_columns]
