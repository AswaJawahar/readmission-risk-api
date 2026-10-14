import streamlit as st

from ui import BASE_DIR, FOOTER, inject_css

st.set_page_config(page_title="Discharge support", page_icon=":material/local_hospital:", layout="wide")
inject_css()
st.logo(str(BASE_DIR / "assets" / "logo.svg"), icon_image=str(BASE_DIR / "assets" / "logo_icon.svg"),
        size="large")

ROLE_LABELS = {"discharge_staff": "Discharge staff", "auditor": "Auditor"}

# The patient opened from the queue (None = show the queue) and the mock signed-in role
st.session_state.setdefault("selected_patient", None)
st.session_state.setdefault("role", "discharge_staff")

about = st.Page("app_pages/about.py", title="About the model", icon=":material/info:")
if st.session_state.role == "auditor":
    pages = [st.Page("app_pages/audit.py", title="Audit log", icon=":material/history:", default=True), about]
else:
    pages = [
        st.Page("app_pages/queue.py", title="Patient queue", icon=":material/groups:", default=True),
        st.Page("app_pages/what_if.py", title="What-if", icon=":material/tune:"),
        about,
    ]
page = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.selectbox("Signed in as (demo)", list(ROLE_LABELS), format_func=ROLE_LABELS.get, key="role")
    st.caption("Mock sign-in for the demo, not real authentication. Discharge staff see patients; "
               "auditors see the audit log.")

page.run()

st.caption(FOOTER)
