import json
import re
from pathlib import Path

import pytest
import requests
import streamlit as st
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

import ui
from main import app as api

api_client = TestClient(api)
ROOT = Path(__file__).resolve().parent.parent
QUEUE = json.loads((ROOT / "demo" / "patient_queue.json").read_text(encoding="utf-8"))["patients"]
TOP_PATIENT = next(p["id"] for p in QUEUE if p["fields"]["number_inpatient"] == 9
                   and p["fields"]["diag_1_group"] == "Missing")
HOSPICE_PATIENT = next(p["id"] for p in QUEUE if p["source"] == "hand_made")


def route(url):
    return "/" + url.split("/", 3)[3]


@pytest.fixture
def app(monkeypatch):
    # Send the app's API calls to the in-process API instead of Render
    monkeypatch.setattr(requests, "post", lambda url, json, timeout: api_client.post(route(url), json=json))
    monkeypatch.setattr(requests, "get", lambda url, timeout: api_client.get(route(url)))
    st.cache_data.clear()
    at = AppTest.from_file("../app.py", default_timeout=60)
    at.run()
    assert not at.exception
    return at


def open_patient(at, patient_id):
    at.session_state["selected_patient"] = patient_id
    at.run()
    assert not at.exception
    return at


def all_text(at):
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [e.value for kind in ("warning", "info", "error", "success") for e in at.get(kind)]
    return "\n".join(parts)


# ---- queue -----------------------------------------------------------------

def test_queue_is_the_main_screen_sorted_by_risk(app):
    assert app.title[0].value == "Patient queue"
    queue = app.dataframe[0].value
    assert len(queue) == len(QUEUE) == 25
    assert queue["Risk tier"].value_counts().to_dict() == {"High": 8, "Moderate": 8, "Low": 8, "Out of scope": 1}
    assert queue.iloc[0]["Patient"] == TOP_PATIENT
    assert queue.iloc[-1]["Risk tier"] == "Out of scope"
    assert queue.iloc[0]["Compared with average"] == "about 7.0× the average"
    assert set(queue.columns).isdisjoint({"_score", "Score"})


def test_tier_filter(app):
    app.pills[0].set_value(["High"]).run()
    queue = app.dataframe[0].value
    assert len(queue) == 8 and set(queue["Risk tier"]) == {"High"}


# ---- dashboard ---------------------------------------------------------------

def test_dashboard_leads_with_tier_and_comparison(app):
    open_patient(app, TOP_PATIENT)
    text = all_text(app)
    assert f"Patient {TOP_PATIENT}" in [s.value for s in app.subheader]
    assert "High risk — about 7.0× the average readmission rate" in [s.value for s in app.subheader]
    assert "(top 10% of patients)" in text
    assert "Score 78% (average: 11%)" in text
    assert "26.7% of high-risk patients were readmitted" in text


def test_dashboard_factors_use_association_wording(app):
    open_patient(app, TOP_PATIENT)
    factors = app.dataframe[0].value
    assert factors.iloc[0]["Factor"] == "9 inpatient stays in the year before this admission"
    assert factors.iloc[0]["Association"] == "Higher score"
    assert set(factors["Association"]) <= {"Higher score", "Lower score", "Little effect"}
    text = all_text(app)
    assert "most associated with a higher score" in text
    assert not re.search(r"\b(raises|lowers) risk\b", text, re.IGNORECASE)


def test_dashboard_shows_cited_considerations_with_review_labels(app):
    open_patient(app, TOP_PATIENT)
    text = all_text(app)
    assert ":gray-badge[ADA-SOC-2026-S16-01]" in text
    assert ":orange-badge[Source pending review]" in text
    assert "Advisory only" in text


def test_preset_question_is_answered(app):
    open_patient(app, TOP_PATIENT)
    app.pills[0].set_value("What education should the patient get?").run()
    text = all_text(app)
    assert "**Q:** What education should the patient get?" in text
    assert "AHRQ-RED-TOOL1-05" in text or "AHRQ-RED-TOOL1-07" in text


def test_out_of_scope_question_is_refused(app):
    open_patient(app, TOP_PATIENT)
    app.text_input[0].input("What insulin dose should I prescribe?")
    next(b for b in app.button if b.label == "Ask").click().run()
    assert any("doesn't advise on diagnoses, medication doses" in w.value for w in app.warning)


def test_hospice_case_shows_the_out_of_scope_guardrail(app):
    open_patient(app, HOSPICE_PATIENT)
    assert any("expired or went to hospice" in w.value for w in app.warning)
    text = all_text(app)
    assert "Hand-made demo case" in text and "Discharge-planning considerations" not in text


def test_back_to_queue(app):
    open_patient(app, TOP_PATIENT)
    next(b for b in app.button if b.label == "Back to queue").click().run()
    assert app.title[0].value == "Patient queue"


# ---- other pages -------------------------------------------------------------

def test_what_if_tab(app):
    app.switch_page("app_pages/what_if.py").run()
    assert not app.exception and app.title[0].value == "What-if"
    assert "Unknown" in next(s for s in app.selectbox if s.label == "Race").options
    assert next(s for s in app.selectbox if s.label == "Gender").options == ["Female", "Male"]
    next(b for b in app.button if b.label == "Show results").click().run()
    text = all_text(app)
    assert "Low risk — below the average readmission rate (about 0.6×)" in [s.value for s in app.subheader]
    assert "Score 7% (average: 11%)" in text


def test_what_if_hospice_disposition_is_out_of_scope(app):
    app.switch_page("app_pages/what_if.py").run()
    discharge = next(s for s in app.selectbox if s.label == "Discharge disposition")
    discharge.select_index(discharge.options.index("Expired")).run()
    next(b for b in app.button if b.label == "Show results").click().run()
    assert any("no prediction is made" in w.value for w in app.warning)


def test_about_page(app):
    app.switch_page("app_pages/about.py").run()
    assert not app.exception
    assert app.metric[0].value == "0.664"
    assert len(app.dataframe[0].value) == 3  # tier table
    assert "Too few readmissions" in set(app.dataframe[1].value["AUC 95% interval"])


# ---- configuration and wording ---------------------------------------------------

def test_api_url_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("API_URL", "https://example.test/api/")
    assert ui.api_url() == "https://example.test/api"
    monkeypatch.delenv("API_URL")
    assert ui.api_url() == ui.DEFAULT_API_URL


def test_comparison_wording():
    assert ui.comparison(0.19, 0.112) == "about 1.7× the average readmission rate"
    assert ui.comparison(0.11, 0.112) == "about the average readmission rate"
    assert ui.comparison(0.067, 0.112) == "below the average readmission rate (about 0.6×)"


def test_no_compliance_claims_in_the_ui():
    for path in [ROOT / "app.py", ROOT / "ui.py", *(ROOT / "app_pages").glob("*.py")]:
        assert "hipaa" not in path.read_text(encoding="utf-8").lower(), path.name
