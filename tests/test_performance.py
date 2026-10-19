"""50-request memory and latency test against a real uvicorn server (not the in-process client).

Measures the server's resident memory (the number that matters for Render's 512 MB free
tier) and per-request latency, and writes the measurements to
testing/results/performance.json for the test report.
"""
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

from test_predict import TEST_SET_TOP_ROW

psutil = pytest.importorskip("psutil")

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "testing" / "results"
REQUESTS = 50
MEMORY_LIMIT_MB = 512          # Render free tier
P95_LIMIT_MS = 2000            # NFR: prediction and explanation "within a few seconds"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def tree_rss_mb(proc):
    procs = [proc] + proc.children(recursive=True)
    return sum(p.memory_info().rss for p in procs if p.is_running()) / 2**20


@pytest.mark.performance
def test_50_requests_memory_and_latency(tmp_path):
    port = free_port()
    env = dict(os.environ, AUDIT_LOG_PATH=str(tmp_path / "audit.jsonl"))
    start = time.perf_counter()
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--port", str(port)],
                              cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc = psutil.Process(server.pid)
    url = f"http://127.0.0.1:{port}"
    try:
        while True:
            try:
                requests.get(url, timeout=1)
                break
            except requests.RequestException:
                assert server.poll() is None, "server exited during startup"
                assert time.perf_counter() - start < 120, "server did not start within 120 s"
                time.sleep(0.1)
        startup_s = time.perf_counter() - start
        rss_after_startup = tree_rss_mb(proc)

        latencies, peak = {"/predict": [], "/explain": [], "/assistant": []}, rss_after_startup
        bodies = {"/predict": TEST_SET_TOP_ROW, "/explain": TEST_SET_TOP_ROW,
                  "/assistant": {"patient": TEST_SET_TOP_ROW}}
        endpoints = list(bodies)
        for i in range(REQUESTS):
            endpoint = endpoints[i % len(endpoints)]
            t = time.perf_counter()
            response = requests.post(f"{url}{endpoint}", json=bodies[endpoint], timeout=30)
            latencies[endpoint].append((time.perf_counter() - t) * 1000)
            assert response.status_code == 200, (endpoint, response.text[:200])
            peak = max(peak, tree_rss_mb(proc))
    finally:
        server.kill()
        server.wait(timeout=15)

    every = [ms for values in latencies.values() for ms in values]
    p95 = statistics.quantiles(every, n=20)[18]
    result = {
        "requests": REQUESTS,
        "startup_seconds": round(startup_s, 2),
        "rss_after_startup_mb": round(rss_after_startup, 1),
        "rss_peak_mb": round(peak, 1),
        "latency_ms": {
            "median_all": round(statistics.median(every), 1),
            "p95_all": round(p95, 1),
            "max_all": round(max(every), 1),
            **{f"median{e.replace('/', '_')}": round(statistics.median(v), 1) for e, v in latencies.items()},
        },
        "limits": {"memory_mb": MEMORY_LIMIT_MB, "p95_ms": P95_LIMIT_MS},
        "measured_on": sys.platform,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "performance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    assert peak < MEMORY_LIMIT_MB
    assert p95 < P95_LIMIT_MS
