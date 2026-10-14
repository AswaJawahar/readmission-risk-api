import json
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from pydantic_core import PydanticCustomError
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()

model = joblib.load(BASE_DIR / "readmission_model.pkl")
model_features = joblib.load(BASE_DIR / "model_features.pkl")
# Allowed values computed from the training split by build_metadata.py
metadata = json.loads((BASE_DIR / "model_metadata.json").read_text(encoding="utf-8"))

NUMERIC_RANGES = metadata["numeric_ranges"]
CATEGORIES = metadata["categories"]
DISCHARGE_CODES = metadata["discharge_codes"]

@app.get("/")
def read_root():
    return {"status": "API is running"}

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

@app.post("/predict")
def predict(patient: PatientData):
    input_dict = patient.model_dump()
    input_df = pd.DataFrame([input_dict])

    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

    risk_score = model.predict_proba(input_encoded)[0][1]

    return {
        "readmission_risk_score": round(float(risk_score), 3),
        "high_risk": bool(risk_score > 0.5)
    }
