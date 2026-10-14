"""Assistant settings, read from environment variables or a gitignored .env file.

Real environment variables win over .env. The API key is only ever read here and
passed to the provider; it is never logged or returned by the API.
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
ROLES = {"discharge_staff", "auditor"}


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "none"       # "none" = rule-based mode
    llm_api_key: str = ""
    llm_model: str = ""
    llm_timeout_seconds: float = 8.0
    kb_require_verified: bool = False  # True: only use excerpts a reviewer has approved
    # Mock role-based access: {"<key>": {"role": "discharge_staff" | "auditor", "user": "<mock id>"}}.
    # Empty = open access (requests are logged as anonymous); /audit then stays unavailable.
    role_keys: dict = field(default_factory=dict, hash=False, compare=True)
    audit_log_path: str = str(BASE_DIR / "logs" / "audit.jsonl")


def parse_role_keys(raw):
    """ROLE_KEYS JSON -> dict; entries with an unknown role or no user are rejected."""
    if not raw or not raw.strip():
        return {}
    keys = json.loads(raw)
    for key, identity in keys.items():
        if identity.get("role") not in ROLES or not identity.get("user"):
            raise ValueError(f"ROLE_KEYS entry for a key has an invalid role or user: {identity}")
    return keys


def read_env_file(path):
    """KEY=VALUE lines; blank lines and # comments ignored; optional quotes stripped."""
    values = {}
    path = Path(path)
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def load_settings(environ=None, env_file=ENV_FILE):
    values = {**read_env_file(env_file), **(os.environ if environ is None else environ)}
    return Settings(
        llm_provider=values.get("LLM_PROVIDER", "none").strip().lower() or "none",
        llm_api_key=values.get("LLM_API_KEY", ""),
        llm_model=values.get("LLM_MODEL", ""),
        llm_timeout_seconds=float(values.get("LLM_TIMEOUT_SECONDS") or 8),
        kb_require_verified=values.get("KB_REQUIRE_VERIFIED", "false").strip().lower() in {"1", "true", "yes"},
        role_keys=parse_role_keys(values.get("ROLE_KEYS", "")),
        audit_log_path=values.get("AUDIT_LOG_PATH") or Settings.audit_log_path,
    )
