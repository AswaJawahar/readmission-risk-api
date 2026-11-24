import streamlit as st

from ui import ROLES, sign_in

# Everything sits in one top-level container, so the first element of the next page
# replaces all of it at once (otherwise the cards linger, faded, while the queue loads)
with st.container():
    st.title("Discharge support")
    st.markdown("Readmission risk and discharge-planning support for patients with diabetes. "
                "Choose your role to continue.")

    # Columns sit side by side on a wide screen and stack on a phone
    for column, (role, info) in zip(st.columns(len(ROLES), border=True), ROLES.items()):
        with column:
            st.subheader(f"{info['icon']} {info['label']}", anchor=False)
            st.markdown(info["description"])
            st.button(f"Continue as {info['label'].lower()}", key=f"sign_in_{role}", type="primary",
                      icon=":material/login:", on_click=sign_in, args=(role,), width="stretch")

    st.info("Mock sign-in for a student proof of concept: choosing a role here is not authentication. "
            "There are no user accounts or passwords. The app calls the API with a shared demo key for each "
            "role, and the API records that role in its audit log.", icon=":material/info:")
