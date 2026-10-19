"""Cold-start test against the live Render API.

Skipped unless RUN_LIVE_TESTS=1, so normal test runs never touch the network. Render's free
tier can put an idle service to sleep; the first request then includes the wake-up time. The
test can't force a cold start, so it records the first-request time and reports whether it
looked cold (over 5 s) or warm. Measurements go to testing/results/live.json.
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
import requests

from test_predict import TEST_SET_TOP_ROW

LIVE_API = os.environ.get("LIVE_API_URL", "https://readmission-risk-api-oyvy.onrender.com")
RESULTS = Path(__file__).resolve().parent.parent / "testing" / "results"
# Allowance for a Render free-tier wake-up. Set when this test was written, before its first run
# (which then measured 71.5 s); a judgement call, not a figure from the requirements.
FIRST_RESPONSE_LIMIT_S = 120
WARM_LIMIT_S = 5               # NFR: results "within a few seconds" once running

pytestmark = pytest.mark.skipif(os.environ.get("RUN_LIVE_TESTS") != "1",
                                reason="live test; set RUN_LIVE_TESTS=1 to run against Render")


@pytest.mark.live
def test_cold_start_live_render_api():
    started = time.perf_counter()
    first = requests.get(f"{LIVE_API}/", timeout=FIRST_RESPONSE_LIMIT_S)
    first_s = time.perf_counter() - started
    assert first.status_code == 200 and first.json() == {"status": "API is running"}

    warm = []
    for _ in range(3):
        t = time.perf_counter()
        response = requests.post(f"{LIVE_API}/predict", json=TEST_SET_TOP_ROW, timeout=30)
        warm.append(time.perf_counter() - t)
        assert response.status_code == 200
        body = response.json()
        assert body["readmission_risk_score"] == 0.782 and body["risk_tier"] == "High"

    t = time.perf_counter()
    explained = requests.post(f"{LIVE_API}/explain", json=TEST_SET_TOP_ROW, timeout=30)
    explain_s = time.perf_counter() - t
    assert explained.status_code == 200

    result = {
        "api": LIVE_API,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "first_response_seconds": round(first_s, 2),
        "looked_cold": first_s > 5,
        "warm_predict_seconds": [round(s, 2) for s in warm],
        "explain_seconds": round(explain_s, 2),
        "limits": {"first_response_s": FIRST_RESPONSE_LIMIT_S, "warm_s": WARM_LIMIT_S},
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "live.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    # Every run is kept: a cold start only shows up when Render was asleep at the time
    with (RESULTS / "live_history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")

    assert first_s < FIRST_RESPONSE_LIMIT_S
    assert max(warm) < WARM_LIMIT_S and explain_s < WARM_LIMIT_S
