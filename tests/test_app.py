import pytest
import requests
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from main import app as api

api_client = TestClient(api)


@pytest.fixture
def form(monkeypatch):
    # Route the form's request to the in-process API instead of Render
    monkeypatch.setattr(requests, "post", lambda url, json: api_client.post("/predict", json=json))
    at = AppTest.from_file("../app.py", default_timeout=30)
    at.run()
    assert not at.exception
    return at


def widget(widgets, label):
    return next(w for w in widgets if w.label == label)


def test_form_offers_only_values_the_api_accepts(form):
    assert widget(form.selectbox, "Gender").options == ["Female", "Male"]
    assert "Unknown" in widget(form.selectbox, "Race").options
    assert widget(form.number_input, "Prior inpatient visits").max == 21


def test_default_prediction(form):
    form.button[0].click().run()
    assert form.metric[0].value == "6.7%"


def test_expired_shows_out_of_scope_message(form):
    widget(form.selectbox, "Discharge disposition").select("Expired").run()
    form.button[0].click().run()
    assert not form.metric
    assert "no prediction is made" in form.warning[0].value
