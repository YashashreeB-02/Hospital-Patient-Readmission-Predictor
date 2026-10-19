import pytest
import pandas as pd
from preprocess import (
    categorize_diagnosis,
    map_discharge,
    map_admission,
    map_source,
    prepare_single_input,
    AGE_ORDER,
    AGE_MAP,
)


def test_categorize_diagnosis():
    # Diabetes ICD-9 250
    assert categorize_diagnosis("250.01") == "Diabetes"
    assert categorize_diagnosis("250") == "Diabetes"

    # Circulatory 390-459 or 785
    assert categorize_diagnosis("410") == "Circulatory"
    assert categorize_diagnosis("785") == "Circulatory"

    # Respiratory 460-519 or 786
    assert categorize_diagnosis("493") == "Respiratory"
    assert categorize_diagnosis("786") == "Respiratory"

    # Digestive 520-579 or 787
    assert categorize_diagnosis("540") == "Digestive"

    # Missing / Unknown
    assert categorize_diagnosis("?") == "Other"
    assert categorize_diagnosis(None) == "Other"
    assert categorize_diagnosis("V45") == "Other"


def test_map_discharge():
    assert map_discharge(1) == "Home"
    assert map_discharge(6) == "Home Health"
    assert map_discharge(8) == "Home Health"
    assert map_discharge(2) == "Transferred Facility"
    assert map_discharge(11) == "Hospice/Expired"
    assert map_discharge(7) == "AMA"
    assert map_discharge(99) == "Other"


def test_map_admission():
    assert map_admission(1) == "Emergency"
    assert map_admission(2) == "Urgent"
    assert map_admission(3) == "Elective"
    assert map_admission(9) == "Other"


def test_map_source():
    assert map_source(7) == "Emergency Room"
    assert map_source(1) == "Physician/Clinic Referral"
    assert map_source(4) == "Transfer"
    assert map_source(99) == "Other"


def test_prepare_single_input():
    sample_input = {
        "age": "[60-70)",
        "time_in_hospital": 4,
        "num_lab_procedures": 40,
        "num_procedures": 1,
        "num_medications": 10,
        "number_outpatient": 1,
        "number_emergency": 1,
        "number_inpatient": 2,
        "number_diagnoses": 7,
        "diag_1_cat": "Circulatory",
        "discharge_clean": "Home",
        "admission_clean": "Emergency",
        "source_clean": "Emergency Room",
        "diabetesMed": "Yes",
        "change": "No",
        "insulin": "Steady",
        "A1Cresult": "None",
        "max_glu_serum": "None",
    }

    feature_cols = [
        "age_num", "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency", "number_inpatient",
        "number_diagnoses", "diag_1_cat", "discharge_clean", "admission_clean", "source_clean",
        "diabetesMed", "change", "insulin", "A1Cresult", "max_glu_serum",
        "total_prior_visits", "service_density", "prior_inpatient_ratio"
    ]

    encoders = {
        "diag_1_cat": {"Circulatory": 0, "Other": 1},
        "discharge_clean": {"Home": 0, "Other": 1},
        "admission_clean": {"Emergency": 0, "Other": 1},
        "source_clean": {"Emergency Room": 0, "Other": 1},
        "diabetesMed": {"Yes": 0, "No": 1},
        "change": {"No": 0, "Ch": 1},
        "insulin": {"Steady": 0, "No": 1},
        "A1Cresult": {"None": 0, ">7": 1},
        "max_glu_serum": {"None": 0, ">200": 1},
    }

    df = prepare_single_input(sample_input, feature_cols, encoders)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert list(df.columns) == feature_cols

    # Verify calculated engineered features
    row = df.iloc[0]
    assert row["age_num"] == AGE_MAP["[60-70)"]
    assert row["total_prior_visits"] == 1 + 1 + 2  # outpatient + emergency + inpatient = 4
    assert row["service_density"] == round((10 + 40) / 4, 2)  # (10 meds + 40 labs) / 4 days = 12.5
    assert row["prior_inpatient_ratio"] == round(2 / (4 + 1), 3)  # 2 inpatient / (4 + 1) = 0.4
