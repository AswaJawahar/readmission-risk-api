from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("readmission_model.pkl")
model_features = joblib.load("model_features.pkl")

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
    
@app.post("/predict")
def predict(patient: PatientData):
    input_dict = patient.dict()
    input_df = pd.DataFrame([input_dict])
    
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)
    
    risk_score = model.predict_proba(input_encoded)[0][1]
    
    return {
        "readmission_risk_score": round(float(risk_score), 3),
        "high_risk": bool(risk_score > 0.5)
    }