import dataclasses
import json

import pytest
from fastapi.testclient import TestClient

import main
from audit import ALLOWED_FIELDS, AuditLog
from config import parse_role_keys
from test_predict import APP_DEFAULTS, TEST_SET_TOP_ROW

client = TestClient(main.app)
STAFF_KEY, AUDITOR_KEY = "staff-test-key", "auditor-test-key"
ROLE_KEYS = {STAFF_KEY: {"role": "discharge_staff", "user": "nurse-01"},
             AUDITOR_KEY: {"role": "auditor", "user": "auditor-01"}}


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, role_keys=ROLE_KEYS))


def entries(log):
    return [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]


# ---- what gets logged ------------------------------------------------------------

def test_predict_is_logged_with_metadata_only(temporary_audit_log):
    response = client.post("/predict", json=TEST_SET_TOP_ROW)
    [entry] = entries(temporary_audit_log)
    assert entry["request_id"] == response.headers["X-Request-ID"]
    assert entry["endpoint"] == "/predict" and entry["status"] == 200
    assert entry["risk_tier"] == "High" and entry["role"] == "anonymous"
    assert set(entry) <= ALLOWED_FIELDS
    text = temporary_audit_log.path.read_text(encoding="utf-8")
    for value in ("AfricanAmerican", "Female", "Neoplasms", "Missing", "[80-90)"):
        assert value not in text


def test_assistant_logs_outcome_and_question_length_not_text(temporary_audit_log):
    question = "What education should the patient get?"
    client.post("/assistant", json={"patient": TEST_SET_TOP_ROW, "question": question})
    client.post("/assistant", json={"patient": TEST_SET_TOP_ROW, "question": "What insulin dose should I prescribe?"})
    answered, refused = entries(temporary_audit_log)
    assert answered["assistant_status"] == "answered" and answered["question_length"] == len(question)
    assert answered["cited_sources"] and answered["assistant_mode"] == "rule_based"
    assert refused["assistant_status"] == "refused"
    text = temporary_audit_log.path.read_text(encoding="utf-8")
    assert "education" not in text and "insulin" not in text


def test_validation_errors_log_types_not_values(temporary_audit_log):
    client.post("/predict", json=dict(APP_DEFAULTS, gender="Femle"))
    client.post("/predict", json=dict(APP_DEFAULTS, discharge_disposition_id=13))
    typo, hospice = entries(temporary_audit_log)
    assert typo["status"] == 422 and typo["error_types"] == ["invalid_category"]
    assert hospice["error_types"] == ["out_of_scope"]
    assert "Femle" not in temporary_audit_log.path.read_text(encoding="utf-8")


def test_unaudited_endpoints_are_not_logged(temporary_audit_log):
    client.get("/")
    client.get("/model-info")
    assert not temporary_audit_log.path.exists()


def test_log_drops_unexpected_fields(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    log.write({"endpoint": "/predict", "race": "Caucasian", "question": "text", "status": 200})
    [entry] = entries(log)
    assert "race" not in entry and "question" not in entry and entry["status"] == 200


# ---- open mode (no ROLE_KEYS: today's deployment) ---------------------------------

def test_open_mode_keeps_staff_endpoints_available_and_audit_closed():
    assert client.post("/explain", json=APP_DEFAULTS).status_code == 200
    assert client.post("/assistant", json={"patient": APP_DEFAULTS}).status_code == 200
    response = client.get("/audit")
    assert response.status_code == 403 and "ROLE_KEYS is not configured" in response.json()["detail"]
    assert client.get("/assistant/config").json()["access"] == "open"


# ---- mock role-based access (ROLE_KEYS set) ----------------------------------------

def test_predict_stays_open_for_backward_compatibility(keys):
    assert client.post("/predict", json=APP_DEFAULTS).status_code == 200


def test_staff_endpoints_need_the_staff_role(keys, temporary_audit_log):
    assert client.post("/explain", json=APP_DEFAULTS).status_code == 401
    assert client.post("/explain", json=APP_DEFAULTS, headers={"X-API-Key": AUDITOR_KEY}).status_code == 403
    ok = client.post("/explain", json=APP_DEFAULTS, headers={"X-API-Key": STAFF_KEY})
    assert ok.status_code == 200
    assert client.post("/assistant", json={"patient": APP_DEFAULTS},
                       headers={"X-API-Key": STAFF_KEY}).status_code == 200
    statuses = [(e["status"], e["role"], e["user"]) for e in entries(temporary_audit_log)]
    assert statuses == [(401, "anonymous", "anonymous"), (403, "auditor", "auditor-01"),
                        (200, "discharge_staff", "nurse-01"), (200, "discharge_staff", "nurse-01")]


def test_unknown_key_is_rejected(keys):
    response = client.post("/predict", json=APP_DEFAULTS, headers={"X-API-Key": "guess"})
    assert response.status_code == 401 and response.json()["detail"] == "Unknown API key."


def test_audit_log_needs_the_auditor_role(keys):
    client.post("/predict", json=APP_DEFAULTS)
    client.post("/predict", json=TEST_SET_TOP_ROW)
    assert client.get("/audit").status_code == 401
    assert client.get("/audit", headers={"X-API-Key": STAFF_KEY}).status_code == 403
    body = client.get("/audit?limit=2", headers={"X-API-Key": AUDITOR_KEY}).json()
    # Newest first; the staff attempt above is logged too
    assert [e["status"] for e in body["entries"]] == [403, 401]
    assert body["total"] == 4


def test_viewing_the_audit_log_is_itself_logged(keys, temporary_audit_log):
    client.get("/audit", headers={"X-API-Key": AUDITOR_KEY})
    [entry] = entries(temporary_audit_log)
    assert entry["endpoint"] == "/audit" and entry["user"] == "auditor-01" and entry["entries_returned"] == 0


def test_keys_never_appear_in_the_log_or_config(keys, temporary_audit_log):
    client.post("/explain", json=APP_DEFAULTS, headers={"X-API-Key": STAFF_KEY})
    client.get("/audit", headers={"X-API-Key": AUDITOR_KEY})
    text = temporary_audit_log.path.read_text(encoding="utf-8")
    config = json.dumps(client.get("/assistant/config").json())
    for key in (STAFF_KEY, AUDITOR_KEY):
        assert key not in text and key not in config


def test_role_keys_are_validated():
    assert parse_role_keys("") == {}
    assert parse_role_keys(json.dumps(ROLE_KEYS)) == ROLE_KEYS
    with pytest.raises(ValueError):
        parse_role_keys(json.dumps({"k": {"role": "admin", "user": "x"}}))
    with pytest.raises(ValueError):
        parse_role_keys(json.dumps({"k": {"role": "auditor"}}))
