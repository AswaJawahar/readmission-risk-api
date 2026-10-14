import streamlit as st

from ui import FOOTER

st.set_page_config(page_title="Discharge support", page_icon=":material/local_hospital:", layout="wide")

# The patient opened from the queue (None = show the queue)
st.session_state.setdefault("selected_patient", None)

page = st.navigation(
    [
        st.Page("app_pages/queue.py", title="Patient queue", icon=":material/groups:", default=True),
        st.Page("app_pages/what_if.py", title="What-if", icon=":material/tune:"),
        st.Page("app_pages/about.py", title="About the model", icon=":material/info:"),
    ],
    position="top",
)
page.run()

st.caption(FOOTER)
