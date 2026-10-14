import pytest

import main
from audit import AuditLog


@pytest.fixture(autouse=True)
def temporary_audit_log(tmp_path, monkeypatch):
    """Tests write audit entries to a temporary file, never to logs/audit.jsonl."""
    log = AuditLog(tmp_path / "audit.jsonl")
    monkeypatch.setattr(main, "audit_log", log)
    return log
