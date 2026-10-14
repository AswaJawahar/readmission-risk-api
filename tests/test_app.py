import dataclasses
import json
import re
from pathlib import Path

import pytest
import requests
import streamlit as st
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

import main
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
    monkeypatch.setattr(requests, "post", lambda url, json, timeout, headers=None:
                        api_client.post(route(url), json=json, headers=headers))
    monkeypatch.setattr(requests, "get", lambda url, timeout, headers=None:
                        api_client.get(route(url), headers=headers))
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


def html_blocks(at):
    """Bodies of st.html blocks (risk card, factor bars, chips), excluding the stylesheet."""
    return [e.proto.body for e in at.get("html") if not e.proto.body.lstrip().startswith("<style>")]


def all_text(at):
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + html_blocks(at)
    parts += [e.value for kind in ("warning", "info", "error", "success") for e in at.get(kind)]
    return "\n".join(parts)


def risk_card(at):
    return next(b for b in html_blocks(at) if "cp-risk" in b)


# ---- queue -----------------------------------------------------------------

def test_queue_is_the_main_screen_sorted_by_risk(app):
    assert app.title[0].value == "Patient queue"
    queue = app.dataframe[0].value
    assert len(queue) == len(QUEUE) == 25
    assert queue["Risk tier"].value_counts().to_dict() == {"High": 8, "Moderate": 8, "Low": 8, "Out of scope": 1}
    assert queue.iloc[0]["Patient"] == TOP_PATIENT
    assert queue.iloc[-1]["Risk tier"] == "Out of scope"
    assert queue.iloc[0]["Compared with average"] == "about 7.0× the average"
    assert queue.iloc[0]["Score"] == pytest.approx(78.2)
    assert "_score" not in queue.columns
    # Short display labels for long discharge destinations
    assert "Rehab facility" in set(queue["Discharge destination"])
    assert not any(len(d) > 30 for d in queue["Discharge destination"])


def test_tier_filter(app):
    app.pills[0].set_value(["High"]).run()
    queue = app.dataframe[0].value
    assert len(queue) == 8 and set(queue["Risk tier"]) == {"High"}


# ---- dashboard ---------------------------------------------------------------

def test_dashboard_leads_with_tier_and_comparison(app):
    open_patient(app, TOP_PATIENT)
    card = risk_card(app)
    assert f"Patient {TOP_PATIENT}" in [s.value for s in app.subheader]
    assert "cp-tier-high" in card and '<span class="cp-badge">High risk</span>' in card
    assert '<span class="cp-multiple-value">7.0×</span>' in card
    # The full sentence is the multiple's hover text; the line under it is just the tier's scope
    assert 'title="High risk — about 7.0× the average readmission rate"' in card
    assert '<div class="cp-scope">Top 10% of patients</div>' in card
    assert "Score 78% (average: 11%)" in card
    assert "26.7% of high-risk patients were readmitted" in card
    assert 'class="cp-pbar-marker" style="left:99%"' in card  # percentile 100, kept inside the bar
    assert "Among the top 1% of test-set scores" in card and "about 100%" not in card


def test_dashboard_factors_use_association_wording(app):
    open_patient(app, TOP_PATIENT)
    bars = next(b for b in html_blocks(app) if "cp-factors" in b)
    first = bars.split('<div class="cp-factor ')[1]
    assert first.startswith("cp-dir-higher") and "9 inpatient stays in the year before this admission" in first
    assert ">Higher score<" in first and 'style="width:55%"' in first
    assert "Discharge destination: Home" in bars  # short display label
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
    card = risk_card(app)
    assert "cp-tier-oos" in card and '<span class="cp-badge">Out of scope</span>' in card
    assert "expired or went to hospice" in card and "No prediction is made for this patient" in card
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
    assert "Low risk — about 0.6× the average readmission rate" in risk_card(app)
    assert "Score 7% (average: 11%)" in text


def test_what_if_hospice_disposition_is_out_of_scope(app):
    app.switch_page("app_pages/what_if.py").run()
    discharge = next(s for s in app.selectbox if s.label == "Discharge disposition")
    discharge.select_index(discharge.options.index("Expired")).run()
    next(b for b in app.button if b.label == "Show results").click().run()
    assert "no prediction is made" in risk_card(app)


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
    assert ui.comparison(0.11, 0.112) == "about 1.0× the average readmission rate"
    assert ui.comparison(0.067, 0.112) == "about 0.6× the average readmission rate"


def test_no_compliance_claims_in_the_ui():
    for path in [ROOT / "app.py", ROOT / "ui.py", *(ROOT / "app_pages").glob("*.py"), *(ROOT / "assets").glob("*")]:
        assert "hipaa" not in path.read_text(encoding="utf-8").lower(), path.name


def test_moderate_note_only_on_moderate_dashboards(app):
    note = "The Moderate tier starts at the median score (10.1%), just below the 11.2% average readmission rate."
    for patient in QUEUE:
        if patient["source"] != "test_set":
            continue
        open_patient(app, patient["id"])
        card = risk_card(app)
        tier = next(t for t in ("High", "Moderate", "Low") if f"cp-tier-{t.lower()}" in card)
        assert (note in card) == (tier == "Moderate"), (patient["id"], tier)
        assert "below the average" not in all_text(app)


# ---- mock sign-in and audit log ------------------------------------------------------

ROLE_KEYS = {"staff-key": {"role": "discharge_staff", "user": "nurse-01"},
             "auditor-key": {"role": "auditor", "user": "auditor-01"}}


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, role_keys=ROLE_KEYS))
    monkeypatch.setenv("STAFF_API_KEY", "staff-key")
    monkeypatch.setenv("AUDITOR_API_KEY", "auditor-key")


def sign_in(at, role):
    at.selectbox(key="role").set_value(role).run()
    assert not at.exception
    return at


def test_staff_dashboard_works_when_keys_are_required(keys, app, temporary_audit_log):
    open_patient(app, TOP_PATIENT)
    assert "High risk — about 7.0× the average readmission rate" in risk_card(app)
    users = {json.loads(line)["user"] for line in temporary_audit_log.path.read_text(encoding="utf-8").splitlines()}
    assert users == {"nurse-01"}


def test_auditor_sees_the_audit_log_not_patients(keys, app):
    open_patient(app, TOP_PATIENT)  # creates some log entries as staff
    sign_in(app, "auditor")
    assert app.title[0].value == "Audit log"
    log = app.dataframe[0].value
    assert {"nurse-01"} <= set(log["user"]) and "/explain" in set(log["endpoint"])
    assert "Patient queue" not in [t.value for t in app.title]


def test_audit_log_page_explains_when_keys_are_not_configured(app):
    sign_in(app, "auditor")
    assert app.title[0].value == "Audit log"
    assert any("isn't available" in i.value for i in app.info)


def test_switching_back_to_staff_shows_the_queue(app):
    sign_in(app, "auditor")
    sign_in(app, "discharge_staff")
    assert app.title[0].value == "Patient queue"


# ---- styling ---------------------------------------------------------------------------

def test_every_discharge_code_has_a_short_label():
    labels = main.metadata["discharge_codes"]["labels"]
    for code in labels:
        short = ui.discharge_short(code, labels)
        assert short in ui.SHORT_DISCHARGE.values() and len(short) <= 34, code


def test_light_and_dark_palettes_define_the_same_colours():
    light, dark = ui.css_vars("cp-light"), ui.css_vars("cp-dark")
    assert set(light) == set(dark) and len(light) > 20


def test_red_is_only_used_for_the_high_tier():
    for theme in ("cp-light", "cp-dark"):
        colours = ui.css_vars(theme)
        high = {v for k, v in colours.items() if k.startswith("high-")}
        assert all(v not in high for k, v in colours.items() if not k.startswith("high-")), theme


def test_queue_tier_cells_use_the_stylesheet_colours():
    light = ui.css_vars("cp-light")
    assert ui.tier_cell_style("High") == (f"background-color: {light['high-bg']}; "
                                          f"color: {light['high-text']}; font-weight: 600")
    assert ui.tier_cell_style("Error") == ""


def test_every_tier_badge_is_text_labelled(app):
    for patient_id in (TOP_PATIENT, HOSPICE_PATIENT):
        open_patient(app, patient_id)
        badge = risk_card(app).split('<span class="cp-badge">')[1].split("</span>")[0]
        assert badge in {"High risk", "Moderate risk", "Low risk", "Out of scope"}


def test_scope_line_is_tier_specific(app):
    expected = {"High": "Top 10% of patients",
                "Moderate": "Between the 50th and 90th percentile of patients",
                "Low": "Lower 50% of patients"}
    seen = set()
    for patient in QUEUE:
        if patient["source"] != "test_set":
            continue
        open_patient(app, patient["id"])
        card = risk_card(app)
        tier = next(t for t in expected if f"cp-tier-{t.lower()}" in card)
        if tier not in seen:
            assert f'<div class="cp-scope">{expected[tier]}</div>' in card, tier
            seen.add(tier)
        if seen == set(expected):
            break
    assert seen == set(expected)
