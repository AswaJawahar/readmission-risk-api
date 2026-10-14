import json
from pathlib import Path

import streamlit as st
import requests

# /explain returns everything /predict does, plus the factors behind the score
API_URL = "https://readmission-risk-api-oyvy.onrender.com/explain"

# Same allowed values and ranges the API validates against (built from the
# training split by build_metadata.py), so the form never offers a value the
# API will reject.
metadata = json.loads((Path(__file__).resolve().parent / "model_metadata.json").read_text(encoding="utf-8"))
ranges = metadata["numeric_ranges"]
categories = metadata["categories"]
tiers = metadata["risk_tiers"]["tiers"]
base_rate = metadata["performance"]["base_rate"]

# Labels come from reference/IDS_mapping.csv. Expired / hospice codes are
# offered too, so the API's out-of-scope message is shown for them.
discharge_labels = {int(code): label for code, label in metadata["discharge_codes"]["labels"].items()}
label_counts = {label: list(discharge_labels.values()).count(label) for label in discharge_labels.values()}


def discharge_label(code):
    # Two codes share "Not recorded" (NULL / Not Mapped); show the code to tell them apart
    label = discharge_labels[code]
    return f"{label} (code {code})" if label_counts[label] > 1 else label


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
    discharge_disposition_id = st.selectbox("Discharge disposition", list(discharge_labels),
                                            format_func=discharge_label)
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
        tier = next(t for t in tiers if t["tier"] == result["risk_tier"])

        st.metric("Readmission Risk Score", f"{score:.1%}")

        tier_text = (
            f"**{tier['tier']} risk** — higher than about {result['risk_percentile']}% of "
            f"test-set patients. In the test set, {tier['readmission_rate']:.1%} of "
            f"{tier['tier'].lower()}-risk patients were readmitted within 30 days "
            f"(overall: {base_rate:.1%})."
        )
        {"High": st.error, "Moderate": st.warning, "Low": st.success}[tier["tier"]](tier_text)

        st.subheader("What drove this score")
        st.write(result["summary"])
        arrows = {"raises": "▲ Raises risk", "lowers": "▼ Lowers risk", "minimal": "• Little effect"}
        for factor in result["factors"][:5]:
            st.markdown(f"{arrows[factor['direction']]} — {factor['description']} "
                        f"({factor['share']:.0%} of the explanation)")
        st.caption("These factors show how the model arrived at this score, not clinical causes. "
                   "Age, gender and race are shown because the model uses them; they are not "
                   "things discharge planning can change.")
    elif response.status_code == 422:
        for error in response.json()["detail"]:
            if error["type"] == "out_of_scope":
                st.warning(error["msg"])
            else:
                st.error(error["msg"])
    else:
        st.error(f"API error: {response.status_code}")