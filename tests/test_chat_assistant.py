import pytest
from unittest.mock import MagicMock, patch
from chat_assistant import (
    is_valid_api_key,
    get_api_config,
    ask_about_prediction,
    draft_followup_message,
)
from data_analytics import (
    calculate_gender_stats,
    calculate_age_stats,
    calculate_medication_stats,
    get_cohort_summary,
    handle_data_query,
)


def test_is_valid_api_key():
    assert is_valid_api_key(None) is False
    assert is_valid_api_key("") is False
    assert is_valid_api_key("short") is False
    assert is_valid_api_key("YOUR_OPENAI_API_KEY") is False
    assert is_valid_api_key("your-openai-api-key-here") is False
    assert is_valid_api_key("sk-placeholder123456789") is False
    assert is_valid_api_key("sk-abcdef1234567890abcdefghijkl") is True


def test_get_api_config_override():
    key, model = get_api_config(override_key="custom-key-12345678", override_model="gpt-4o")
    assert key == "custom-key-12345678"
    assert model == "gpt-4o"


def test_cohort_summary_calculations():
    summary = get_cohort_summary()
    assert summary["total_encounters"] == 99343
    assert summary["readmitted_count"] == 46816
    assert summary["readmission_rate_percent"] == 47.13
    assert summary["readmit_under_30_count"] == 11314
    assert summary["readmit_over_30_count"] == 35502
    assert summary["no_readmit_count"] == 52527


def test_gender_calculations():
    stats = calculate_gender_stats()
    assert "Male" in stats
    assert "Female" in stats
    assert stats["Male"]["total_encounters"] == 45886
    assert stats["Male"]["readmitted_count"] == 21195
    assert stats["Male"]["readmission_rate_percent"] == 46.19
    assert stats["Female"]["total_encounters"] == 53454
    assert stats["Female"]["readmitted_count"] == 25621
    assert stats["Female"]["readmission_rate_percent"] == 47.93


def test_age_calculations():
    age_list = calculate_age_stats()
    assert len(age_list) == 10
    # Top risk age group is [80-90)
    top_age = age_list[0]
    assert top_age["age_bracket"] == "[80-90)"
    assert top_age["readmission_rate_percent"] == 50.36
    # Lowest is [0-10)
    lowest_age = age_list[-1]
    assert lowest_age["age_bracket"] == "[0-10)"
    assert lowest_age["readmission_rate_percent"] == 18.12


def test_medication_calculations():
    med_stats = calculate_medication_stats()
    # Insulin
    ins = {item["regimen"]: item for item in med_stats["insulin"]}
    assert ins["Down"]["rate_percent"] == 54.06
    assert ins["Up"]["rate_percent"] == 52.96
    assert ins["Steady"]["rate_percent"] == 46.19
    assert ins["No"]["rate_percent"] == 44.57


def test_data_query_engine_queries():
    # 1. Gender query
    handled, resp, data = handle_data_query("How many male and female patients are in the dataset?")
    assert handled is True
    assert "45,886" in resp
    assert "53,454" in resp

    # 2. Male risk query
    handled, resp, data = handle_data_query("How many male candidates are there in risk?")
    assert handled is True
    assert "21,195" in resp
    assert "46.19%" in resp

    # 3. Overall rate
    handled, resp, data = handle_data_query("What percentage of patients were readmitted?")
    assert handled is True
    assert "47.13%" in resp
    assert "46,816" in resp

    # 4. Age groups
    handled, resp, data = handle_data_query("Which age groups have the highest readmission rates?")
    assert handled is True
    assert "[80-90)" in resp
    assert "50.36%" in resp

    # 5. Unavailable feature
    handled, resp, data = handle_data_query("What is patient blood pressure?")
    assert handled is True
    assert "Unavailable in Dataset" in resp


def test_data_query_followup():
    history = [
        {"role": "user", "content": "How many male candidates are there in risk?"},
        {"role": "assistant", "content": "There are 45,886 male encounters in the dataset..."}
    ]
    handled, resp, data = handle_data_query("What about females?", chat_history=history)
    assert handled is True
    assert "Female Encounters Focus" in resp
    assert "25,621" in resp
    assert "47.93%" in resp


def test_patient_specific_vs_dataset():
    context = {
        "readmission_probability_percent": 60.1,
        "risk_category": "High Risk",
        "existing_rule_based_flags": ["⚠️ Frequent Hospitalizations: 2 prior stays"],
        "patient_inputs": {
            "age": "[60-70)",
            "time_in_hospital": 7,
            "diabetesMed": "Yes",
            "insulin": "Up",
            "discharge_clean": "Home Health",
            "diag_1_cat": "Circulatory"
        }
    }

    # Patient-specific question -> handled by patient fallback
    ans_patient = ask_about_prediction(context, "Why was this risk level assigned to this patient?")
    assert "Risk Stratification Summary & Clinical Drivers" in ans_patient
    assert "High Risk" in ans_patient

    # Dataset question -> handled by data analytics engine
    ans_dataset = ask_about_prediction(context, "How many male candidates are there in risk?")
    assert "Gender Demographic & Readmission Breakdown" in ans_dataset
    assert "21,195" in ans_dataset
    assert "45,886" in ans_dataset

    # Ensure responses are completely different
    assert ans_patient != ans_dataset


def test_local_fallback_draft():
    context = {
        "readmission_probability_percent": 64.2,
        "risk_category": "High Risk",
        "patient_inputs": {
            "time_in_hospital": 5,
            "discharge_clean": "Home Health",
            "insulin": "Up",
        },
    }

    draft = draft_followup_message(context)
    assert "CLINICAL OUTREACH DRAFT" in draft
    assert "High Risk" in draft
    assert "NOT SENT AUTOMATICALLY" in draft
    assert "[Patient Name]" in draft
    assert "CLINICIAN PRE-SEND VERIFICATION CHECKLIST" in draft
