import os
import pytest
import joblib
import pandas as pd
from preprocess import prepare_single_input


@pytest.fixture(scope="module")
def model_bundle():
    model_path = "readmission_model.pkl"
    assert os.path.exists(model_path), f"Model file {model_path} must exist"
    bundle = joblib.load(model_path)
    return bundle


def test_bundle_structure(model_bundle):
    assert isinstance(model_bundle, dict)
    assert "model" in model_bundle
    assert "feature_columns" in model_bundle
    assert "encoders" in model_bundle
    assert "metrics" in model_bundle

    # Check metrics recorded
    metrics = model_bundle["metrics"]
    assert "test_accuracy" in metrics
    assert "roc_auc" in metrics
    assert 0.60 <= metrics["test_accuracy"] <= 0.70
    assert 0.65 <= metrics["roc_auc"] <= 0.75


def test_inference_pipeline(model_bundle):
    sample_input = {
        "age": "[70-80)",
        "time_in_hospital": 5,
        "num_lab_procedures": 45,
        "num_procedures": 1,
        "num_medications": 18,
        "number_outpatient": 0,
        "number_emergency": 1,
        "number_inpatient": 2,
        "number_diagnoses": 9,
        "diag_1_cat": "Circulatory",
        "discharge_clean": "Home Health",
        "admission_clean": "Emergency",
        "source_clean": "Emergency Room",
        "diabetesMed": "Yes",
        "change": "Ch",
        "insulin": "Up",
        "A1Cresult": ">8",
        "max_glu_serum": "None",
    }

    feature_cols = model_bundle["feature_columns"]
    encoders = model_bundle["encoders"]
    model = model_bundle["model"]

    X = prepare_single_input(sample_input, feature_cols, encoders)
    assert X.shape == (1, len(feature_cols))

    probabilities = model.predict_proba(X)[0]
    prediction = model.predict(X)[0]

    assert len(probabilities) == 2
    assert 0.0 <= probabilities[0] <= 1.0
    assert 0.0 <= probabilities[1] <= 1.0
    assert pytest.approx(probabilities[0] + probabilities[1], 0.001) == 1.0
    assert prediction in [0, 1]
