import pytest
import pandas as pd
from data_analytics import (
    get_cohort_summary,
    calculate_gender_stats,
    calculate_age_stats,
    calculate_medication_stats,
    calculate_diagnosis_stats,
    calculate_lab_stats,
    calculate_utilization_stats,
    calculate_discharge_stats,
    calculate_admission_stats,
    handle_data_query,
)


@pytest.fixture
def sample_synthetic_dataset():
    """
    Independent 10-row synthetic test fixture with known ground truth:
    - Total: 10 encounters
    - Gender: 5 Female, 4 Male, 1 Unknown/Invalid
    - Readmission outcomes:
        - 3 '<30' (readmitted)
        - 2 '>30' (readmitted)
        - 5 'NO'  (not readmitted)
        => Total readmitted = 5 / 10 = 50.00%
    - Gender distribution:
        - Female: 5 encounters, 3 readmitted (60.00%)
        - Male: 4 encounters, 2 readmitted (50.00%)
        - Unknown/Invalid: 1 encounter, 0 readmitted (0.00%)
    - Age brackets:
        - [80-90): 2 encounters, 2 readmitted (100.00%)
        - [70-80): 3 encounters, 2 readmitted (66.67%)
        - [60-70): 3 encounters, 1 readmitted (33.33%)
        - [30-40): 2 encounters, 0 readmitted (0.00%)
    - Insulin regimens:
        - Down: 2 encounters, 2 readmitted (100.00%)
        - Up: 2 encounters, 1 readmitted (50.00%)
        - Steady: 3 encounters, 1 readmitted (33.33%)
        - No: 3 encounters, 1 readmitted (33.33%)
    - Medication adjustments:
        - Ch: 4 encounters, 3 readmitted (75.00%)
        - No: 6 encounters, 2 readmitted (33.33%)
    """
    rows = [
        # Female, [80-90), <30 (readmitted), Down, Ch, Yes, 8 days, 1 inpatient, 0 emergency
        {
            "encounter_id": 1, "gender": "Female", "age": "[80-90)", "readmitted": "<30",
            "insulin": "Down", "change": "Ch", "diabetesMed": "Yes", "time_in_hospital": 8,
            "number_inpatient": 1, "number_emergency": 0, "num_medications": 20, "num_lab_procedures": 50,
            "diag_1": "250.0", "discharge_disposition_id": 1, "admission_type_id": 1, "admission_source_id": 7,
            "A1Cresult": ">8", "max_glu_serum": "None"
        },
        # Female, [80-90), >30 (readmitted), Down, Ch, Yes, 9 days, 2 inpatient, 1 emergency
        {
            "encounter_id": 2, "gender": "Female", "age": "[80-90)", "readmitted": ">30",
            "insulin": "Down", "change": "Ch", "diabetesMed": "Yes", "time_in_hospital": 9,
            "number_inpatient": 2, "number_emergency": 1, "num_medications": 18, "num_lab_procedures": 45,
            "diag_1": "414.0", "discharge_disposition_id": 1, "admission_type_id": 1, "admission_source_id": 7,
            "A1Cresult": ">7", "max_glu_serum": "None"
        },
        # Female, [70-80), <30 (readmitted), Up, Ch, Yes, 7 days, 1 inpatient, 0 emergency
        {
            "encounter_id": 3, "gender": "Female", "age": "[70-80)", "readmitted": "<30",
            "insulin": "Up", "change": "Ch", "diabetesMed": "Yes", "time_in_hospital": 7,
            "number_inpatient": 1, "number_emergency": 0, "num_medications": 16, "num_lab_procedures": 40,
            "diag_1": "428.0", "discharge_disposition_id": 6, "admission_type_id": 2, "admission_source_id": 1,
            "A1Cresult": "Norm", "max_glu_serum": "None"
        },
        # Female, [70-80), NO (not readmitted), Steady, No, Yes, 3 days, 0 inpatient, 0 emergency
        {
            "encounter_id": 4, "gender": "Female", "age": "[70-80)", "readmitted": "NO",
            "insulin": "Steady", "change": "No", "diabetesMed": "Yes", "time_in_hospital": 3,
            "number_inpatient": 0, "number_emergency": 0, "num_medications": 12, "num_lab_procedures": 30,
            "diag_1": "486", "discharge_disposition_id": 1, "admission_type_id": 3, "admission_source_id": 1,
            "A1Cresult": "None", "max_glu_serum": "None"
        },
        # Female, [60-70), NO (not readmitted), No, No, No, 2 days, 0 inpatient, 0 emergency
        {
            "encounter_id": 5, "gender": "Female", "age": "[60-70)", "readmitted": "NO",
            "insulin": "No", "change": "No", "diabetesMed": "No", "time_in_hospital": 2,
            "number_inpatient": 0, "number_emergency": 0, "num_medications": 8, "num_lab_procedures": 20,
            "diag_1": "715", "discharge_disposition_id": 1, "admission_type_id": 3, "admission_source_id": 1,
            "A1Cresult": "None", "max_glu_serum": "None"
        },
        # Male, [70-80), >30 (readmitted), Up, No, Yes, 5 days, 1 inpatient, 0 emergency
        {
            "encounter_id": 6, "gender": "Male", "age": "[70-80)", "readmitted": ">30",
            "insulin": "Up", "change": "No", "diabetesMed": "Yes", "time_in_hospital": 5,
            "number_inpatient": 1, "number_emergency": 0, "num_medications": 14, "num_lab_procedures": 35,
            "diag_1": "250.0", "discharge_disposition_id": 1, "admission_type_id": 1, "admission_source_id": 7,
            "A1Cresult": "None", "max_glu_serum": ">200"
        },
        # Male, [60-70), <30 (readmitted), Steady, No, Yes, 4 days, 0 inpatient, 1 emergency
        {
            "encounter_id": 7, "gender": "Male", "age": "[60-70)", "readmitted": "<30",
            "insulin": "Steady", "change": "No", "diabetesMed": "Yes", "time_in_hospital": 4,
            "number_inpatient": 0, "number_emergency": 1, "num_medications": 15, "num_lab_procedures": 42,
            "diag_1": "410.0", "discharge_disposition_id": 1, "admission_type_id": 1, "admission_source_id": 7,
            "A1Cresult": ">8", "max_glu_serum": "None"
        },
        # Male, [60-70), NO (not readmitted), Steady, No, Yes, 3 days, 0 inpatient, 0 emergency
        {
            "encounter_id": 8, "gender": "Male", "age": "[60-70)", "readmitted": "NO",
            "insulin": "Steady", "change": "No", "diabetesMed": "Yes", "time_in_hospital": 3,
            "number_inpatient": 0, "number_emergency": 0, "num_medications": 10, "num_lab_procedures": 28,
            "diag_1": "584", "discharge_disposition_id": 1, "admission_type_id": 2, "admission_source_id": 1,
            "A1Cresult": "None", "max_glu_serum": "None"
        },
        # Male, [30-40), NO (not readmitted), No, No, No, 1 day, 0 inpatient, 0 emergency
        {
            "encounter_id": 9, "gender": "Male", "age": "[30-40)", "readmitted": "NO",
            "insulin": "No", "change": "No", "diabetesMed": "No", "time_in_hospital": 1,
            "number_inpatient": 0, "number_emergency": 0, "num_medications": 5, "num_lab_procedures": 15,
            "diag_1": "820", "discharge_disposition_id": 1, "admission_type_id": 1, "admission_source_id": 7,
            "A1Cresult": "None", "max_glu_serum": "None"
        },
        # Unknown/Invalid, [30-40), NO (not readmitted), No, No, No, 2 days, 0 inpatient, 0 emergency
        {
            "encounter_id": 10, "gender": "Unknown/Invalid", "age": "[30-40)", "readmitted": "NO",
            "insulin": "No", "change": "No", "diabetesMed": "No", "time_in_hospital": 2,
            "number_inpatient": 0, "number_emergency": 0, "num_medications": 6, "num_lab_procedures": 18,
            "diag_1": "786", "discharge_disposition_id": 1, "admission_type_id": 1, "admission_source_id": 7,
            "A1Cresult": "None", "max_glu_serum": "None"
        },
    ]
    return pd.DataFrame(rows)


def test_independent_fixture_cohort_summary(sample_synthetic_dataset):
    """Test get_cohort_summary against ground-truth expected values in synthetic fixture."""
    summary = get_cohort_summary(sample_synthetic_dataset)
    assert summary["total_encounters"] == 10
    assert summary["readmitted_count"] == 5
    assert summary["readmission_rate_percent"] == 50.0
    assert summary["readmit_under_30_count"] == 3
    assert summary["readmit_under_30_percent"] == 30.0
    assert summary["readmit_over_30_count"] == 2
    assert summary["readmit_over_30_percent"] == 20.0
    assert summary["no_readmit_count"] == 5
    assert summary["no_readmit_percent"] == 50.0


def test_independent_fixture_gender_calculations(sample_synthetic_dataset):
    """Test calculate_gender_stats against ground-truth expected counts and rates."""
    stats = calculate_gender_stats(sample_synthetic_dataset)
    assert len(stats) == 3
    assert "Female" in stats
    assert "Male" in stats
    assert "Unknown/Invalid" in stats

    # Female: 5 total, 3 readmitted (60.0%), 2 not readmitted
    assert stats["Female"]["total_encounters"] == 5
    assert stats["Female"]["readmitted_count"] == 3
    assert stats["Female"]["readmission_rate_percent"] == 60.0
    assert stats["Female"]["not_readmitted_count"] == 2
    assert stats["Female"]["percent_of_cohort"] == 50.0

    # Male: 4 total, 2 readmitted (50.0%), 2 not readmitted
    assert stats["Male"]["total_encounters"] == 4
    assert stats["Male"]["readmitted_count"] == 2
    assert stats["Male"]["readmission_rate_percent"] == 50.0
    assert stats["Male"]["not_readmitted_count"] == 2
    assert stats["Male"]["percent_of_cohort"] == 40.0

    # Unknown/Invalid: 1 total, 0 readmitted (0.0%), 1 not readmitted
    assert stats["Unknown/Invalid"]["total_encounters"] == 1
    assert stats["Unknown/Invalid"]["readmitted_count"] == 0
    assert stats["Unknown/Invalid"]["readmission_rate_percent"] == 0.0
    assert stats["Unknown/Invalid"]["not_readmitted_count"] == 1
    assert stats["Unknown/Invalid"]["percent_of_cohort"] == 10.0


def test_independent_fixture_age_calculations(sample_synthetic_dataset):
    """Test calculate_age_stats ranking and rate computations against ground truth."""
    age_list = calculate_age_stats(sample_synthetic_dataset)
    assert len(age_list) == 4

    # Top risk bracket: [80-90) with 2/2 = 100%
    assert age_list[0]["age_bracket"] == "[80-90)"
    assert age_list[0]["total_encounters"] == 2
    assert age_list[0]["readmitted_count"] == 2
    assert age_list[0]["readmission_rate_percent"] == 100.0

    # Second: [70-80) with 2/3 = 66.67%
    assert age_list[1]["age_bracket"] == "[70-80)"
    assert age_list[1]["total_encounters"] == 3
    assert age_list[1]["readmitted_count"] == 2
    assert age_list[1]["readmission_rate_percent"] == 66.67

    # Lowest: [30-40) with 0/2 = 0.0%
    assert age_list[-1]["age_bracket"] == "[30-40)"
    assert age_list[-1]["total_encounters"] == 2
    assert age_list[-1]["readmitted_count"] == 0
    assert age_list[-1]["readmission_rate_percent"] == 0.0


def test_independent_fixture_medications_and_insulin(sample_synthetic_dataset):
    """Test calculate_medication_stats against ground-truth insulin and change numbers."""
    med_stats = calculate_medication_stats(sample_synthetic_dataset)
    ins_map = {item["regimen"]: item for item in med_stats["insulin"]}

    # Down: 2/2 = 100%
    assert ins_map["Down"]["total"] == 2
    assert ins_map["Down"]["readmitted"] == 2
    assert ins_map["Down"]["rate_percent"] == 100.0

    # Up: 2/2 = 100%
    assert ins_map["Up"]["total"] == 2
    assert ins_map["Up"]["readmitted"] == 2
    assert ins_map["Up"]["rate_percent"] == 100.0

    # Steady: 1/3 = 33.33%
    assert ins_map["Steady"]["total"] == 3
    assert ins_map["Steady"]["readmitted"] == 1
    assert ins_map["Steady"]["rate_percent"] == 33.33

    # No: 0/3 = 0.0%
    assert ins_map["No"]["total"] == 3
    assert ins_map["No"]["readmitted"] == 0
    assert ins_map["No"]["rate_percent"] == 0.0

    # Change
    ch = med_stats["change"]
    assert ch["Ch"]["total"] == 3
    assert ch["Ch"]["readmitted"] == 3
    assert ch["Ch"]["rate_percent"] == 100.0

    assert ch["No"]["total"] == 7
    assert ch["No"]["readmitted"] == 2
    assert ch["No"]["rate_percent"] == 28.57


def test_empty_dataset_handling():
    """Verify that calculations on an empty dataset return safe zero values without ZeroDivisionError."""
    empty_df = pd.DataFrame(columns=["gender", "age", "readmitted", "insulin", "change", "diabetesMed", "discharge_disposition_id", "admission_type_id", "time_in_hospital", "number_inpatient", "number_emergency"])

    summary = get_cohort_summary(empty_df)
    assert summary["total_encounters"] == 0
    assert summary["readmission_rate_percent"] == 0.0

    g_stats = calculate_gender_stats(empty_df)
    assert g_stats == {}

    age_stats = calculate_age_stats(empty_df)
    assert age_stats == []

    med_stats = calculate_medication_stats(empty_df)
    assert med_stats["insulin"] == []

    handled, resp, data = handle_data_query("How many male patients are there?", df=empty_df)
    assert handled is True
    assert "0 records" in resp


def test_missing_values_graceful_handling():
    """Verify that datasets with missing/null values do not crash analytics."""
    nan_df = pd.DataFrame([
        {"gender": "Female", "age": "[60-70)", "readmitted": "NO", "insulin": None, "change": None, "discharge_disposition_id": 1, "time_in_hospital": 2},
        {"gender": "Male", "age": None, "readmitted": "<30", "insulin": "Steady", "change": "No", "discharge_disposition_id": 1, "time_in_hospital": 3},
    ])
    g_stats = calculate_gender_stats(nan_df)
    assert len(g_stats) == 2
    assert g_stats["Female"]["total_encounters"] == 1
    assert g_stats["Male"]["total_encounters"] == 1


def test_handle_data_query_synthetic_fixture(sample_synthetic_dataset):
    """Test handle_data_query producing dynamically verified markdown using the synthetic fixture."""
    # Gender query on fixture
    handled, resp, data = handle_data_query("How many male and female patients are in the dataset?", df=sample_synthetic_dataset)
    assert handled is True
    assert "Female" in resp
    assert "Male" in resp
    assert "10" in resp  # Total encounters in fixture is 10
    assert "60.00%" in resp  # Female readmission rate in fixture

    # Age query on fixture
    handled, resp, data = handle_data_query("Which age groups have the highest readmission rates?", df=sample_synthetic_dataset)
    assert handled is True
    assert "[80-90)" in resp
    assert "100.00%" in resp


def test_unsupported_questions_transparent_boundary():
    """Verify that unrecorded clinical variables are honestly reported with available alternatives."""
    unsupported = [
        ("What is the patient blood pressure?", "Blood Pressure"),
        ("Do you have smoking status?", "Smoking"),
        ("What is the average patient BMI?", "Bmi"),
        ("What are the hospital charges and costs?", "Cost"),
        ("Which doctor treated the patient?", "Doctor"),
        ("What is patient cholesterol level?", "Cholesterol"),
    ]
    for q, expected_var in unsupported:
        handled, resp, data = handle_data_query(q)
        assert handled is True
        assert "Unavailable in Dataset" in resp
        assert "Available Variables in `diabetic_data.csv`" in resp
        assert "Recommended Alternative" in resp


def test_differently_worded_questions():
    """Verify that different natural-language phrasings route to the correct calculation engine."""
    queries_and_keywords = [
        ("What is the sex distribution of readmitted patients?", "Gender Demographic"),
        ("How many women were readmitted?", "Gender Demographic"),
        ("Readmission rates by age bracket", "Readmission Rate by Age Group"),
        ("Which elderly patients are at highest risk?", "Readmission Rate by Age Group"),
        ("How does insulin titration affect readmission?", "Pharmacotherapy & Insulin Impact"),
        ("Does changing medications increase readmission risk?", "Pharmacotherapy & Insulin Impact"),
        ("What is the overall readmission rate across the entire hospital cohort?", "Overall Cohort Readmission Metrics"),
        ("What percentage of encounters were readmitted under 30 days?", "Overall Cohort Readmission Metrics"),
        ("How do primary diagnoses relate to readmission rates?", "Primary Diagnosis Category"),
        ("How does discharge destination affect readmission?", "Discharge Destination Category"),
        ("What is the difference between emergency and elective admissions?", "Hospital Admission Type"),
        ("How many patients had a hospital stay of 7 days or more?", "Extended Hospital Stays"),
    ]
    for q, keyword in queries_and_keywords:
        handled, resp, data = handle_data_query(q)
        assert handled is True, f"Failed to handle query: {q}"
        assert keyword in resp, f"Expected '{keyword}' in response for query: '{q}'"


def test_followup_question_resolution():
    """Verify that follow-up questions retain context from preceding conversation turns."""
    history = [
        {"role": "user", "content": "How many male patients were readmitted?"},
        {"role": "assistant", "content": "There are 45,886 male encounters in the dataset..."}
    ]
    handled, resp, data = handle_data_query("What about females?", chat_history=history)
    assert handled is True
    assert "Female Encounters Focus" in resp
    assert "25,621" in resp
    assert "47.93%" in resp

    # Age follow-up
    age_history = [
        {"role": "user", "content": "Which age groups have the highest readmission rates?"},
        {"role": "assistant", "content": "Patients in the [80-90) bracket had the highest readmission rate..."}
    ]
    handled, resp, data = handle_data_query("What about 70-80 years?", chat_history=age_history)
    assert handled is True
    assert "Readmission Rate by Age Group" in resp
    assert "[70-80)" in resp


def test_unrecognized_cohort_question_safety():
    """Verify that unrecognized cohort questions do not fall into patient-specific answers."""
    handled, resp, data = handle_data_query("What are the statistical correlations across all patients in the database?")
    assert handled is True
    assert "Cohort Query Guidance" in resp
    assert "Available Features" in resp
