import json
import re

import pytest
from fastapi.testclient import TestClient

import build_patient_queue
from main import app, metadata
from test_predict import APP_DEFAULTS

client = TestClient(app)
queue = json.loads(build_patient_queue.QUEUE_PATH.read_text(encoding="utf-8"))
patients = queue["patients"]


def test_queue_has_fake_ids_and_only_the_14_fields():
    ids = [p["id"] for p in patients]
    assert len(ids) == 25 and len(set(ids)) == 25
    assert all(re.fullmatch(r"PT-\d{4}", i) for i in ids)
    for p in patients:
        assert set(p["fields"]) == set(APP_DEFAULTS)
        assert set(p) <= {"id", "source", "note", "fields"}
    text = json.dumps(queue)
    assert "encounter_id" not in text and "patient_nbr" not in text


def test_one_hand_made_out_of_scope_case():
    hand_made = [p for p in patients if p["source"] == "hand_made"]
    assert len(hand_made) == 1 and "not a dataset record" in hand_made[0]["note"]
    assert hand_made[0]["fields"]["discharge_disposition_id"] in metadata["discharge_codes"]["out_of_scope"]
    response = client.post("/predict", json=hand_made[0]["fields"])
    assert response.status_code == 422 and response.json()["detail"][0]["type"] == "out_of_scope"


def test_test_set_patients_cover_every_tier():
    tiers = [client.post("/predict", json=p["fields"]).json()["risk_tier"]
             for p in patients if p["source"] == "test_set"]
    assert {t: tiers.count(t) for t in set(tiers)} == {"Low": 8, "Moderate": 8, "High": 8}


@pytest.mark.skipif(not build_patient_queue.CSV_PATH.exists(),
                    reason="clean_diabetic_data.csv is gitignored; only available locally")
def test_committed_queue_can_be_rebuilt():
    assert build_patient_queue.QUEUE_PATH.read_text(encoding="utf-8") == \
        build_patient_queue.to_json(build_patient_queue.build())
