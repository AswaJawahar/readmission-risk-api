import os

import pytest

# Real environment variables override .env, so this keeps a developer's .env (with a real
# LLM key) from being picked up by main at import: tests never call a real LLM provider.
os.environ["LLM_PROVIDER"] = "none"
os.environ["LLM_API_KEY"] = ""

import main  # noqa: E402
from audit import AuditLog


@pytest.fixture(autouse=True)
def temporary_audit_log(tmp_path, monkeypatch):
    """Tests write audit entries to a temporary file, never to logs/audit.jsonl."""
    log = AuditLog(tmp_path / "audit.jsonl")
    monkeypatch.setattr(main, "audit_log", log)
    return log
