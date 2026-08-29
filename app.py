import streamlit as st
import requests

API_URL = "https://readmission-risk-api-oyvy.onrender.com/predict"

st.set_page_config(page_title="Readmission Risk Predictor", page_icon="🏥")
st.title("🏥 30-Day Readmission Risk Predictor")
st.write("Enter patient details to estimate readmission risk, powered by a live Gradient Boosting model.")

col1, col2 = st.columns(2)

with col1:
    number_inpatient = st.number_input("Prior inpatient visits", min_value=0, max_value=20, value=0)
    discharge_disposition_id = st.number_input("Discharge disposition code", min_value=1, max_value=30, value=1)
    number_emergency = st.number_input("Prior emergency visits", min_value=0, max_value=20, value=0)
    number_diagnoses = st.number_input("Number of diagnoses", min_value=1, max_value=20, value=5)
    num_medications = st.number_input("Number of medications", min_value=0, max_value=50, value=10)
    time_in_hospital = st.number_input("Days in hospital", min_value=1, max_value=14, value=3)
    num_lab_procedures = st.number_input("Lab procedures performed", min_value=0, max_value=150, value=40)

with col2:
    diabetesMed = st.selectbox("On diabetes medication?", ["Yes", "No"])
    age = st.selectbox("Age range", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
                                       "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])
    diag_groups = ["Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
                   "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Missing"]
    diag_1_group = st.selectbox("Primary diagnosis category", diag_groups)
    diag_2_group = st.selectbox("Secondary diagnosis category", diag_groups)
    diag_3_group = st.selectbox("Tertiary diagnosis category", diag_groups)

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
    else:
        st.error(f"API error: {response.status_code}")