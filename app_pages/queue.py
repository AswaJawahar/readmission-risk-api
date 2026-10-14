import json

import pandas as pd
import streamlit as st

from ui import (comparison, describe_patient, error_message, error_type, get, load_queue, render_dashboard,
                score_queue)

patients = load_queue()
by_id = {p["id"]: p for p in patients}
selected = st.session_state.selected_patient
# A new table key after each visit, so the previous row selection is not carried over
# (otherwise clicking that row again would deselect it instead of opening the patient)
st.session_state.setdefault("queue_version", 0)
table_key = f"queue_table_{st.session_state.queue_version}"


def close_patient():
    st.session_state.selected_patient = None
    st.session_state.queue_version += 1


def open_selected():
    rows = st.session_state[table_key].selection.rows
    if rows:
        st.session_state.selected_patient = st.session_state.queue_ids[rows[0]]


if selected in by_id:
    patient = by_id[selected]
    st.button("Back to queue", icon=":material/arrow_back:", on_click=close_patient)
    source = ("Hand-made demo case (not a dataset record)" if patient["source"] == "hand_made"
              else "De-identified test-set encounter")
    render_dashboard(f"Patient {patient['id']}", source, patient["fields"], key=patient["id"])
    st.stop()

st.title("Patient queue")
info = get("/model-info")
base_rate = info["performance"]["base_rate"]
labels = info["discharge_codes"]["labels"]

with st.spinner("Scoring the queue..."):
    results = score_queue(json.dumps(patients, sort_keys=True))

rows = []
for patient, (status, body) in zip(patients, results):
    fields = patient["fields"]
    if status == 200:
        tier, score = body["risk_tier"], body["readmission_risk_score"]
        compared = comparison(score, base_rate).replace(" readmission rate", "")
    elif error_type(body) == "out_of_scope":
        tier, score, compared = "Out of scope", None, "No prediction (expired / hospice)"
    else:
        tier, score, compared = "Error", None, error_message(body)
    rows.append({
        "Patient": patient["id"],
        "Risk tier": tier,
        "Compared with average": compared,
        "Age group": describe_patient(fields, labels)["Age group"],
        "Primary diagnosis": "Not recorded" if fields["diag_1_group"] == "Missing" else fields["diag_1_group"],
        "Discharge destination": labels.get(str(fields["discharge_disposition_id"]), "Unknown"),
        "Days in hospital": fields["time_in_hospital"],
        "Prior inpatient stays": fields["number_inpatient"],
        "_score": -1 if score is None else score,
    })

queue = pd.DataFrame(rows).sort_values("_score", ascending=False)
counts = queue["Risk tier"].value_counts()
tiers = [t for t in ("High", "Moderate", "Low", "Out of scope", "Error") if t in counts]
shown_tiers = st.pills("Show tiers", tiers, selection_mode="multi", default=tiers,
                       format_func=lambda t: f"{t} ({counts[t]})", key="tier_filter")
queue = queue[queue["Risk tier"].isin(shown_tiers)].drop(columns="_score").reset_index(drop=True)
st.session_state.queue_ids = queue["Patient"].tolist()

st.caption(f"{len(patients)} demo patients, highest risk first. Select a row to open the dashboard. "
           f"Average = the test-set 30-day readmission rate ({base_rate:.1%}).")
# Tall enough to show every row without an inner scrollbar
st.dataframe(queue, hide_index=True, key=table_key, on_select=open_selected,
             selection_mode="single-row", height=35 * (len(queue) + 1) + 3)
