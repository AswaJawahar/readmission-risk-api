import pandas as pd
import streamlit as st

from ui import audit_entries

st.title("Audit log")
st.caption("Metadata only: who used which endpoint, when, and the outcome. No patient details, "
           "question text, IP addresses or keys are recorded.")

st.button("Refresh", icon=":material/refresh:")
status, body = audit_entries()

if status != 200:
    detail = body.get("detail", "") if isinstance(body, dict) else ""
    st.info(f"The audit log isn't available: {detail} The API needs ROLE_KEYS with an auditor key, and this "
            "app needs that key as AUDITOR_API_KEY (environment or Streamlit secrets).",
            icon=":material/lock:")
    st.stop()

entries = pd.DataFrame(body["entries"])
if entries.empty:
    st.info("No requests have been logged yet.", icon=":material/history:")
    st.stop()

with st.container(horizontal=True):
    st.metric("Requests shown", f"{len(entries):,}", border=True,
              help=f"Newest first; {body['total']:,} entries in the log in total.")
    st.metric("Errors (HTTP 4xx/5xx)", f"{int((entries['status'] >= 400).sum()):,}", border=True)
    if "assistant_status" in entries:
        st.metric("Assistant refusals", f"{int((entries['assistant_status'] == 'refused').sum()):,}", border=True)
        st.metric("Insufficient information",
                  f"{int((entries['assistant_status'] == 'insufficient_information').sum()):,}", border=True)

columns = ["time", "user", "role", "endpoint", "status", "risk_tier", "assistant_status", "assistant_mode",
           "cited_sources", "question_length", "error_types", "latency_ms", "request_id"]
shown = entries.reindex(columns=[c for c in columns if c in entries])
for list_column in ("cited_sources", "error_types"):
    if list_column in shown:
        shown[list_column] = shown[list_column].apply(lambda v: ", ".join(v) if isinstance(v, list) else None)
st.dataframe(shown, hide_index=True, placeholder="—", column_config={
    "time": st.column_config.TextColumn("Time (UTC)"),
    "user": "User", "role": "Role", "endpoint": "Endpoint", "status": "Status",
    "risk_tier": "Risk tier", "assistant_status": "Assistant outcome", "assistant_mode": "Assistant mode",
    "cited_sources": "Cited sources", "error_types": "Error types", "request_id": "Request ID",
    "latency_ms": st.column_config.NumberColumn("Latency (ms)"),
    "question_length": st.column_config.NumberColumn("Question length"),
})
st.caption("Logs are written to the API server's disk. On Render's free tier they are lost when the "
           "service restarts or redeploys.")
