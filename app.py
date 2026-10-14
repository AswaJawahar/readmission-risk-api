import json
from pathlib import Path

import streamlit as st
import requests

API_URL = "https://readmission-risk-api-oyvy.onrender.com/predict"

# Same allowed values and ranges the API validates against (built from the
# training split by build_metadata.py), so the form never offers a value the
# API will reject.
metadata = json.loads((Path(__file__).resolve().parent / "model_metadata.json").read_text(encoding="utf-8"))
ranges = metadata["numeric_ranges"]
categories = metadata["categories"]

# Labels from the UCI dataset's IDs_mapping. 11/13/14 (expired / hospice) stay
# in the list so the API's out-of-scope message is shown for them.
discharge_labels = {
    1: "Discharged to home",
    2: "Discharged/transferred to another short term hospital",
    3: "Discharged/transferred to SNF",
    4: "Discharged/transferred to ICF",
    5: "Discharged/transferred to another type of inpatient care institution",
    6: "Discharged/transferred to home with home health service",
    7: "Left AMA",
    8: "Discharged/transferred to home under care of Home IV provider",
    11: "Expired",
    13: "Hospice / home",
    14: "Hospice / medical facility",
    15: "Discharged/transferred within this institution to Medicare approved swing bed",
    18: "NULL",
    22: "Discharged/transferred to another rehab facility",
    23: "Discharged/transferred to a long term care hospital",
    25: "Not Mapped",
    28: "Discharged/transferred/referred to a psychiatric hospital or psychiatric unit",
}
offered_codes = set(metadata["discharge_codes"]["allowed"]) | set(metadata["discharge_codes"]["out_of_scope"])
discharge_options = {k: v for k, v in discharge_labels.items() if k in offered_codes}


def ordered(preferred, allowed):
    """Keep the form's preferred order, limited to (and completed by) the allowed values."""
    return [v for v in preferred if v in allowed] + [v for v in allowed if v not in preferred]


def number_input(label, field, value):
    return st.number_input(label, min_value=ranges[field]["min"], max_value=ranges[field]["max"], value=value)

st.set_page_config(page_title="Readmission Risk Predictor", page_icon="🏥")
st.title("🏥 30-Day Readmission Risk Predictor")
st.write("Enter patient details to estimate readmission risk, powered by a live Gradient Boosting model.")

col1, col2 = st.columns(2)

with col1:
    number_inpatient = number_input("Prior inpatient visits", "number_inpatient", 0)
    discharge_label = st.selectbox("Discharge disposition", list(discharge_options.values()))
    discharge_disposition_id = [k for k, v in discharge_options.items() if v == discharge_label][0]
    number_emergency = number_input("Prior emergency visits", "number_emergency", 0)
    number_diagnoses = number_input("Number of diagnoses", "number_diagnoses", 5)
    num_medications = number_input("Number of medications", "num_medications", 15)
    time_in_hospital = number_input("Days in hospital", "time_in_hospital", 4)
    num_lab_procedures = number_input("Lab procedures performed", "num_lab_procedures", 43)

with col2:
    diabetesMed = st.selectbox("On diabetes medication?", ordered(["Yes", "No"], categories["diabetesMed"]))
    age = st.selectbox("Age range", categories["age"])
    gender = st.selectbox("Gender", ordered(["Female", "Male"], categories["gender"]))
    race = st.selectbox("Race", ordered(["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"],
                                        categories["race"]))
    diag_groups = ["Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
                   "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Missing"]
    diag_1_group = st.selectbox("Primary diagnosis category", ordered(diag_groups, categories["diag_1_group"]))
    diag_2_group = st.selectbox("Secondary diagnosis category", ordered(diag_groups, categories["diag_2_group"]))
    diag_3_group = st.selectbox("Tertiary diagnosis category", ordered(diag_groups, categories["diag_3_group"]))

if st.button("Predict Readmission Risk", type="primary"):
    payload = {
        "number_inpatient": number_inpatient,
        "discharge_disposition_id": discharge_disposition_id,
        "number_emergency": number_emergency,
        "number_diagnoses": number_diagnoses,
        "num_medications": num_medications,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "diabetesMed": diabetesMed,
        "diag_1_group": diag_1_group,
        "diag_2_group": diag_2_group,
        "diag_3_group": diag_3_group,
        "age": age,
        "gender": gender,
        "race": race,
    }

    with st.spinner("Contacting model..."):
        response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        score = result["readmission_risk_score"]

        st.metric("Readmission Risk Score", f"{score:.1%}")

        if result["high_risk"]:
            st.error("⚠️ High Risk — flagged for readmission risk")
        else:
            st.success("✅ Lower Risk")
    elif response.status_code == 422:
        for error in response.json()["detail"]:
            if error["type"] == "out_of_scope":
                st.warning(error["msg"])
            else:
                st.error(error["msg"])
    else:
        st.error(f"API error: {response.status_code}")