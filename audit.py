"""Metadata-only audit log (JSON lines) and mock role-based access.

Each entry records who called which endpoint, when, the outcome and a few
non-identifying facts about the result (risk tier, assistant status, cited
source IDs, question length). It never records patient field values, question
text, IP addresses or keys.
"""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException

# Entry keys allowed in the log; anything else an endpoint tries to add is dropped
ALLOWED_FIELDS = {
    "time", "request_id", "method", "endpoint", "status", "latency_ms", "role", "user",
    "model_version", "risk_tier", "assistant_mode", "assistant_status", "cited_sources",
    "question_length", "error_types", "entries_returned",
}
ANONYMOUS = {"role": "anonymous", "user": "anonymous"}


class AuditLog:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, entry):
        entry = {k: v for k, v in entry.items() if k in ALLOWED_FIELDS and v is not None}
        entry.setdefault("time", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def latest(self, limit):
        """The newest `limit` entries, newest first, and the total count."""
        if not self.path.exists():
            return [], 0
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        entries = [json.loads(line) for line in lines if line.strip()]
        return entries[::-1][:limit], len(entries)


def identify(api_key, role_keys):
    """Maps an X-API-Key header to a mock identity.

    Returns (identity, error). With no keys configured, everyone is anonymous.
    With keys configured, a missing key is anonymous and an unknown key is an error.
    """
    if not role_keys or not api_key:
        return dict(ANONYMOUS), None
    identity = role_keys.get(api_key)
    if identity is None:
        return dict(ANONYMOUS), "Unknown API key."
    return {"role": identity["role"], "user": identity["user"]}, None


def require_role(request, role, role_keys):
    """Raises 401/403 unless the caller has `role`.

    Without configured keys, staff endpoints stay open (demo mode) but the audit
    log, which needs an auditor, is unavailable.
    """
    identity = request.state.identity
    if not role_keys:
        if role == "auditor":
            raise HTTPException(403, "The audit log needs an auditor key; ROLE_KEYS is not configured.")
        return
    if identity["role"] == "anonymous":
        raise HTTPException(401, "An X-API-Key header is required for this endpoint.")
    if identity["role"] != role:
        raise HTTPException(403, f"This endpoint needs the {role} role.")
