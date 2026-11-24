import streamlit as st

from ui import BASE_DIR, FOOTER, ROLES, inject_css, sign_out

st.set_page_config(page_title="Discharge support", page_icon=":material/local_hospital:", layout="wide")
inject_css()
st.logo(str(BASE_DIR / "assets" / "logo.svg"), icon_image=str(BASE_DIR / "assets" / "logo_icon.svg"),
        size="large")

# The patient opened from the queue (None = show the queue) and the mock signed-in role
# (None = show the sign-in page)
st.session_state.setdefault("selected_patient", None)
st.session_state.setdefault("role", None)
role = st.session_state.role

about = st.Page("app_pages/about.py", title="About the model", icon=":material/info:")
if role is None:
    pages = [st.Page("app_pages/sign_in.py", title="Sign in", icon=":material/login:", default=True)]
elif role == "auditor":
    pages = [st.Page("app_pages/audit.py", title="Audit log", icon=":material/history:", default=True), about]
else:
    pages = [
        st.Page("app_pages/queue.py", title="Patient queue", icon=":material/groups:", default=True),
        st.Page("app_pages/what_if.py", title="What-if", icon=":material/tune:"),
        about,
    ]
page = st.navigation(pages, position="sidebar")

if role is not None:
    with st.sidebar:
        st.markdown(f"Signed in as **{ROLES[role]['label']}** (demo)")
        st.button("Switch role", key="switch_role", icon=":material/swap_horiz:", on_click=sign_out,
                  width="stretch")
        st.caption("Mock sign-in, not real authentication. Discharge staff see patients; "
                   "auditors see the audit log.")

page.run()

st.caption(FOOTER)
