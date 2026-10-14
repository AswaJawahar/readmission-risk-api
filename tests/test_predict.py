import pytest
from fastapi.testclient import TestClient

from main import app, metadata, model_features

client = TestClient(app)

# Default inputs of the Streamlit form
APP_DEFAULTS = {
    "number_inpatient": 0, "discharge_disposition_id": 1, "number_emergency": 0,
    "number_diagnoses": 5, "num_medications": 15, "time_in_hospital": 4,
    "num_lab_procedures": 43, "diabetesMed": "Yes", "diag_1_group": "Circulatory",
    "diag_2_group": "Circulatory", "diag_3_group": "Circulatory", "age": "[0-10)",
    "gender": "Female", "race": "Caucasian",
}
# Highest-scoring encounter in the test set (exploration.ipynb)
TEST_SET_TOP_ROW = dict(
    APP_DEFAULTS, number_inpatient=9, num_medications=21, number_diagnoses=9,
    time_in_hospital=2, num_lab_procedures=2, age="[80-90)", race="AfricanAmerican",
    diag_1_group="Missing", diag_2_group="Neoplasms", diag_3_group="Neoplasms",
)


def post(**overrides):
    return client.post("/predict", json=dict(APP_DEFAULTS, **overrides))


def error_types(response):
    return [e["type"] for e in response.json()["detail"]]


def test_root():
    assert client.get("/").json() == {"status": "API is running"}


def test_valid_request_keeps_original_fields_and_score():
    response = post()
    assert response.status_code == 200
    body = response.json()
    # Original fields unchanged; new fields are additions only
    assert body["readmission_risk_score"] == 0.067 and body["high_risk"] is False
    assert set(body) == {"readmission_risk_score", "high_risk", "risk_tier",
                         "risk_percentile", "model_version"}
    assert body["risk_tier"] == "Low"
    assert body["model_version"] == metadata["model_version"]


def test_test_set_top_row_reproduces_notebook_score():
    body = client.post("/predict", json=TEST_SET_TOP_ROW).json()
    assert body["readmission_risk_score"] == 0.782 and body["high_risk"] is True
    assert body["risk_tier"] == "High" and body["risk_percentile"] == 100


def test_misspelled_category_is_rejected_with_allowed_values():
    response = post(gender="Femle")
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["type"] == "invalid_category"
    assert error["loc"] == ["body", "gender"]
    assert "'Femle' is not a valid value for gender" in error["msg"]
    assert "Female, Male" in error["msg"]


def test_rare_training_category_is_rejected():
    assert "Unknown/Invalid" in metadata["excluded_rare_categories"]["gender"]
    assert error_types(post(gender="Unknown/Invalid")) == ["invalid_category"]


@pytest.mark.parametrize("code", [11, 13, 14, 19, 20, 21])
def test_expired_or_hospice_discharge_is_out_of_scope(code):
    response = post(discharge_disposition_id=code)
    assert response.status_code == 422
    assert error_types(response) == ["out_of_scope"]
    assert "no prediction is made" in response.json()["detail"][0]["msg"]


@pytest.mark.parametrize("code", [9, 24, 99])
def test_rare_or_unseen_discharge_code_is_rejected(code):
    assert error_types(post(discharge_disposition_id=code)) == ["invalid_category"]


@pytest.mark.parametrize("field", list(metadata["numeric_ranges"]))
def test_numeric_range_boundaries(field):
    bounds = metadata["numeric_ranges"][field]
    assert post(**{field: bounds["min"]}).status_code == 200
    assert post(**{field: bounds["max"]}).status_code == 200
    assert error_types(post(**{field: bounds["min"] - 1})) == ["out_of_range"]
    assert error_types(post(**{field: bounds["max"] + 1})) == ["out_of_range"]


def test_wrong_type_and_missing_field_still_rejected():
    assert error_types(post(number_inpatient="many")) == ["int_parsing"]
    payload = dict(APP_DEFAULTS)
    del payload["race"]
    assert error_types(client.post("/predict", json=payload)) == ["missing"]


@pytest.mark.parametrize("field", list(metadata["categories"]))
def test_every_allowed_category_reaches_the_model(field):
    # Each allowed value either has its own model column or is the single
    # drop_first baseline, so no accepted value is silently discarded.
    allowed = metadata["categories"][field]
    without_column = [v for v in allowed if f"{field}_{v}" not in model_features]
    assert len(without_column) == 1
