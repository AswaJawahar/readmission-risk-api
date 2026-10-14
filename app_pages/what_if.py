import streamlit as st

from ui import get, render_dashboard

st.title("What-if")
st.caption("Enter or adjust a patient's details to see how the score, factors and suggestions change. "
           "Options and limits match the values the model was trained on.")

info = get("/model-info")
ranges = info["numeric_ranges"]
categories = info["categories"]
labels = {int(code): label for code, label in info["discharge_codes"]["labels"].items()}
label_counts = {label: list(labels.values()).count(label) for label in labels.values()}


def discharge_label(code):
    # Two codes share "Not recorded" (NULL / Not Mapped); show the code to tell them apart
    label = labels[code]
    return f"{label} (code {code})" if label_counts[label] > 1 else label


def ordered(preferred, allowed):
    """Keep the form's preferred order, limited to (and completed by) the allowed values."""
    return [v for v in preferred if v in allowed] + [v for v in allowed if v not in preferred]


def number_input(label, field, value):
    return st.number_input(label, min_value=ranges[field]["min"], max_value=ranges[field]["max"],
                           value=value, key=f"whatif_{field}")


diag_groups = ["Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
               "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Missing"]

with st.form("what_if_form"):
    col1, col2 = st.columns(2)
    with col1:
        fields = {
            "number_inpatient": number_input("Prior inpatient visits", "number_inpatient", 0),
            "discharge_disposition_id": st.selectbox("Discharge disposition", list(labels),
                                                     format_func=discharge_label, key="whatif_discharge"),
            "number_emergency": number_input("Prior emergency visits", "number_emergency", 0),
            "number_diagnoses": number_input("Number of diagnoses", "number_diagnoses", 5),
            "num_medications": number_input("Number of medications", "num_medications", 15),
            "time_in_hospital": number_input("Days in hospital", "time_in_hospital", 4),
            "num_lab_procedures": number_input("Lab procedures performed", "num_lab_procedures", 43),
        }
    with col2:
        fields.update({
            "diabetesMed": st.selectbox("On diabetes medication?",
                                        ordered(["Yes", "No"], categories["diabetesMed"]), key="whatif_med"),
            "age": st.selectbox("Age range", categories["age"], key="whatif_age"),
            "gender": st.selectbox("Gender", ordered(["Female", "Male"], categories["gender"]),
                                   key="whatif_gender"),
            "race": st.selectbox("Race", ordered(["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"],
                                                 categories["race"]), key="whatif_race"),
            "diag_1_group": st.selectbox("Primary diagnosis category",
                                         ordered(diag_groups, categories["diag_1_group"]), key="whatif_diag1"),
            "diag_2_group": st.selectbox("Secondary diagnosis category",
                                         ordered(diag_groups, categories["diag_2_group"]), key="whatif_diag2"),
            "diag_3_group": st.selectbox("Tertiary diagnosis category",
                                         ordered(diag_groups, categories["diag_3_group"]), key="whatif_diag3"),
        })
    if st.form_submit_button("Show results", type="primary", icon=":material/play_arrow:"):
        # Keep the scenario so the dashboard survives reruns (e.g. asking the assistant)
        st.session_state.what_if_fields = fields

if "what_if_fields" in st.session_state:
    render_dashboard("What-if scenario", "Entered by hand", st.session_state.what_if_fields, key="whatif")
