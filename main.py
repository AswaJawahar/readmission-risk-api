import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from pydantic_core import PydanticCustomError
import joblib
import numpy as np
import pandas as pd
import shap

from assistant import Assistant, make_provider
from audit import AuditLog, identify, require_role
from config import load_settings
from explain import explain as explain_factors, summarize
from knowledge import KnowledgeBase
from risk import assign_tier, percentile_rank

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()

model = joblib.load(BASE_DIR / "readmission_model.pkl")
model_features = joblib.load(BASE_DIR / "model_features.pkl")
explainer = shap.TreeExplainer(model)
# The model's log-odds before any patient-specific contribution (a 1-element array in shap)
BASELINE_LOG_ODDS = float(np.ravel(explainer.expected_value)[0])
knowledge = KnowledgeBase()
settings = load_settings()
assistant = Assistant(knowledge, settings, *make_provider(settings))
audit_log = AuditLog(settings.audit_log_path)
# Endpoints recorded in the audit log (viewing the log is recorded too)
AUDITED_PATHS = {"/predict", "/explain", "/assistant", "/audit"}
# Allowed values computed from the training split by build_metadata.py
metadata = json.loads((BASE_DIR / "model_metadata.json").read_text(encoding="utf-8"))

NUMERIC_RANGES = metadata["numeric_ranges"]
CATEGORIES = metadata["categories"]
DISCHARGE_CODES = metadata["discharge_codes"]
TIER_CUTOFFS = metadata["risk_tiers"]["cutoffs"]
SCORE_QUANTILES = metadata["risk_tiers"]["score_quantiles"]

@app.middleware("http")
async def audit_requests(request: Request, call_next):
    request.state.request_id = uuid.uuid4().hex[:12]
    request.state.audit = {}
    identity, auth_error = identify(request.headers.get("X-API-Key"), settings.role_keys)
    request.state.identity = identity
    start = time.perf_counter()
    status = 500
    try:
        if auth_error:
            response = JSONResponse({"detail": auth_error}, status_code=401)
        else:
            response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request.state.request_id
        return response
    finally:
        if request.url.path in AUDITED_PATHS:
            audit_log.write({
                "request_id": request.state.request_id, "method": request.method,
                "endpoint": request.url.path, "status": status,
                "latency_ms": round((time.perf_counter() - start) * 1000),
                **identity, **request.state.audit,
            })

@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # Only the error types are logged, never the rejected values
    request.state.audit["error_types"] = sorted({e["type"] for e in exc.errors()})
    return await request_validation_exception_handler(request, exc)

@app.get("/")
def read_root():
    return {"status": "API is running"}

@app.get("/model-info")
def model_info():
    # Test-set metrics, tier definitions and allowed inputs, all from build_metadata.py
    return metadata

@app.get("/sources")
def sources():
    # Curated knowledge-base sources and how many excerpts still await human review
    return knowledge.source_list()

class PatientData(BaseModel):
    number_inpatient: int
    discharge_disposition_id: int
    number_emergency: int
    number_diagnoses: int
    num_medications: int
    time_in_hospital: int
    num_lab_procedures: int
    diabetesMed: str
    diag_1_group: str
    diag_2_group: str
    diag_3_group: str
    age: str
    gender: str
    race: str

    @field_validator(*NUMERIC_RANGES)
    @classmethod
    def check_range(cls, value, info):
        bounds = NUMERIC_RANGES[info.field_name]
        if not bounds["min"] <= value <= bounds["max"]:
            raise PydanticCustomError(
                "out_of_range",
                "{field} must be between {min} and {max} (the range seen in training data); got {value}",
                {"field": info.field_name, "value": value, **bounds},
            )
        return value

    @field_validator(*CATEGORIES)
    @classmethod
    def check_category(cls, value, info):
        allowed = CATEGORIES[info.field_name]
        if value not in allowed:
            raise PydanticCustomError(
                "invalid_category",
                "'{value}' is not a valid value for {field}. Allowed values: {allowed}",
                {"field": info.field_name, "value": value, "allowed": ", ".join(allowed)},
            )
        return value

    @field_validator("discharge_disposition_id")
    @classmethod
    def check_discharge_code(cls, value):
        if value in DISCHARGE_CODES["out_of_scope"]:
            raise PydanticCustomError(
                "out_of_scope",
                "Discharge disposition {value} ({label}) means the patient expired or went to "
                "hospice. These encounters were excluded from training because 30-day "
                "readmission isn't clinically possible, so no prediction is made.",
                {"value": value, "label": DISCHARGE_CODES["labels"][str(value)]},
            )
        if value not in DISCHARGE_CODES["allowed"]:
            raise PydanticCustomError(
                "invalid_category",
                "Discharge disposition {value} is not supported (it is missing from the training "
                "data or has too few encounters). Allowed codes: {allowed}",
                {"value": value, "allowed": ", ".join(map(str, DISCHARGE_CODES["allowed"]))},
            )
        return value

def encode(input_dict):
    input_df = pd.DataFrame([input_dict])

    input_encoded = pd.get_dummies(input_df)
    return input_encoded.reindex(columns=model_features, fill_value=0)

def score_response(input_encoded):
    risk_score = model.predict_proba(input_encoded)[0][1]

    return {
        "readmission_risk_score": round(float(risk_score), 3),
        # Legacy flag kept for existing callers; risk_tier is the recommended output
        "high_risk": bool(risk_score > 0.5),
        "risk_tier": assign_tier(risk_score, TIER_CUTOFFS),
        "risk_percentile": percentile_rank(risk_score, SCORE_QUANTILES),
        "model_version": metadata["model_version"],
    }

@app.post("/predict")
def predict(patient: PatientData, request: Request):
    result = score_response(encode(patient.model_dump()))
    request.state.audit.update(model_version=result["model_version"], risk_tier=result["risk_tier"])
    return result

def explain(patient: PatientData):
    input_dict = patient.model_dump()
    input_encoded = encode(input_dict)
    shap_row = explainer.shap_values(input_encoded.astype(float))[0]
    factors = explain_factors(input_dict, shap_row, model_features, DISCHARGE_CODES["labels"])

    return {
        **score_response(input_encoded),
        "summary": summarize(factors),
        "factors": factors,
        "baseline_log_odds": round(BASELINE_LOG_ODDS, 4),
        "method": ("SHAP (TreeExplainer) contributions to the model's log-odds, summed per input "
                   "field. They show how the model arrived at this score, not clinical causes."),
    }

@app.post("/explain")
def explain_endpoint(patient: PatientData, request: Request):
    require_role(request, "discharge_staff", settings.role_keys)
    result = explain(patient)
    request.state.audit.update(model_version=result["model_version"], risk_tier=result["risk_tier"])
    return result

class AssistantRequest(BaseModel):
    patient: PatientData
    question: str | None = Field(default=None, max_length=500)

@app.post("/assistant")
def assistant_reply(body: AssistantRequest, request: Request):
    require_role(request, "discharge_staff", settings.role_keys)
    reply = assistant.respond(explain(body.patient), body.question)
    request.state.audit.update(
        model_version=metadata["model_version"], risk_tier=reply["risk"]["risk_tier"],
        assistant_mode=reply["mode"], assistant_status=reply["status"],
        cited_sources=sorted(reply["citations"]),
        question_length=len(body.question) if body.question else None,
    )
    return reply

@app.get("/audit")
def audit(request: Request, limit: int = Query(100, ge=1, le=1000)):
    # Metadata-only entries, newest first; needs the auditor role
    require_role(request, "auditor", settings.role_keys)
    entries, total = audit_log.latest(limit)
    request.state.audit["entries_returned"] = len(entries)
    return {"entries": entries, "total": total}

@app.get("/assistant/config")
def assistant_config():
    # Which mode is active; the API key itself is never returned
    return {
        "llm_provider": settings.llm_provider,
        "active_mode": "llm" if assistant.provider else "rule_based",
        "note": assistant.note,
        "has_api_key": bool(settings.llm_api_key),
        "kb_require_verified": settings.kb_require_verified,
        "access": "role_keys" if settings.role_keys else "open",
    }
