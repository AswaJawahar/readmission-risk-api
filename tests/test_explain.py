import math
import time

import pytest
from fastapi.testclient import TestClient

from explain import LABELS, describe
from main import app, encode, model, metadata
from test_predict import APP_DEFAULTS, TEST_SET_TOP_ROW

client = TestClient(app)

CASES = [APP_DEFAULTS, TEST_SET_TOP_ROW,
         dict(APP_DEFAULTS, discharge_disposition_id=3, age="[70-80)", race="Unknown"),
         dict(TEST_SET_TOP_ROW, gender="Male", diabetesMed="No", number_emergency=4)]


def explain(payload):
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    return response.json()


@pytest.mark.parametrize("payload", CASES)
def test_contributions_add_up_to_the_model_output(payload):
    body = explain(payload)
    log_odds = model.decision_function(encode(payload))[0]
    total = body["baseline_log_odds"] + sum(f["contribution_log_odds"] for f in body["factors"])
    # Values are rounded to 4 decimals in the response
    assert total == pytest.approx(log_odds, abs=1e-3)
    assert 1 / (1 + math.exp(-log_odds)) == pytest.approx(body["readmission_risk_score"], abs=5e-4)


@pytest.mark.parametrize("payload", CASES)
def test_factors_are_exactly_the_14_inputs(payload):
    factors = explain(payload)["factors"]
    assert sorted(f["field"] for f in factors) == sorted(APP_DEFAULTS)
    assert all(f["value"] == payload[f["field"]] for f in factors)
    assert sum(f["share"] for f in factors) == pytest.approx(1, abs=1e-3)
    sizes = [abs(f["contribution_log_odds"]) for f in factors]
    assert sizes == sorted(sizes, reverse=True)


def test_explain_includes_the_predict_response():
    body = explain(APP_DEFAULTS)
    predicted = client.post("/predict", json=APP_DEFAULTS).json()
    assert {k: body[k] for k in predicted} == predicted


def test_top_row_is_driven_by_prior_inpatient_stays():
    body = explain(TEST_SET_TOP_ROW)
    top = body["factors"][0]
    assert top["field"] == "number_inpatient" and top["direction"] == "raises"
    assert body["summary"].startswith(
        "The factors raising this patient's risk the most are 9 inpatient stays")


def test_direction_matches_sign_and_size():
    for f in explain(TEST_SET_TOP_ROW)["factors"]:
        if f["share"] < 0.02:
            assert f["direction"] == "minimal"
        else:
            assert f["direction"] == ("raises" if f["contribution_log_odds"] > 0 else "lowers")


def test_demographic_fields_are_marked():
    marked = {f["field"] for f in explain(APP_DEFAULTS)["factors"] if f["demographic"]}
    assert marked == {"age", "gender", "race"}


def test_descriptions_use_plain_language():
    labels = metadata["discharge_codes"]["labels"]
    assert describe("number_inpatient", 1, labels) == "1 inpatient stay in the year before this admission"
    assert describe("number_diagnoses", 1, labels) == "1 diagnosis recorded"
    assert describe("discharge_disposition_id", 18, labels) == "Discharge destination: Not recorded"
    assert describe("diag_1_group", "Missing", labels) == "Primary diagnosis: not recorded"
    assert describe("age", "[80-90)", labels) == "Age group: 80-89"
    assert set(LABELS) == set(APP_DEFAULTS)


def test_explain_validates_input():
    response = client.post("/explain", json=dict(APP_DEFAULTS, gender="Femle"))
    assert response.status_code == 422


def test_explain_is_fast():
    explain(APP_DEFAULTS)
    start = time.perf_counter()
    for _ in range(10):
        explain(TEST_SET_TOP_ROW)
    assert (time.perf_counter() - start) / 10 < 0.5
