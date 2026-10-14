"""Shared pieces of the Streamlit app: API client, formatting and the patient dashboard.

Every number shown comes from the API (the live model and model_metadata.json).
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_API_URL = "https://readmission-risk-api-oyvy.onrender.com"
QUEUE_PATH = BASE_DIR / "demo" / "patient_queue.json"

TIER_BADGE = {
    "High": {"color": "red", "icon": ":material/priority_high:"},
    "Moderate": {"color": "orange", "icon": ":material/trending_up:"},
    "Low": {"color": "green", "icon": ":material/check:"},
    "Out of scope": {"color": "gray", "icon": ":material/block:"},
}
ASSOCIATION = {"higher": "Higher score", "lower": "Lower score", "minimal": "Little effect"}
PRESET_QUESTIONS = [
    "What should we do about medications before discharge?",
    "What education should the patient get?",
    "How soon should follow-up happen?",
]
FOOTER = ("Advisory decision support for a student proof of concept. Built on de-identified UCI data "
          "(1999–2008) plus one invented demo case. Not a diagnostic tool; a clinician must review "
          "every result.")


# ---- API client ------------------------------------------------------------

def api_url():
    """API_URL from the environment or Streamlit secrets, else the deployed API."""
    url = os.environ.get("API_URL")
    if not url:
        try:
            url = st.secrets.get("API_URL")
        except Exception:  # no secrets file
            url = None
    return (url or DEFAULT_API_URL).rstrip("/")


def _post(path, payload):
    try:
        response = requests.post(f"{api_url()}{path}", json=payload, timeout=90)
    except requests.RequestException as exc:
        return 0, {"detail": [{"type": "connection", "msg": f"Could not reach the API: {exc}"}]}
    return response.status_code, response.json()


def _get(path):
    response = requests.get(f"{api_url()}{path}", timeout=90)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl="10m", max_entries=200, show_spinner=False)
def post(path, payload_json):
    """Cached POST; the payload is passed as JSON text so it hashes reliably."""
    return _post(path, json.loads(payload_json))


@st.cache_data(ttl="1h", show_spinner=False)
def get(path):
    return _get(path)


def explain(fields):
    return post("/explain", json.dumps(fields, sort_keys=True))


def ask(fields, question=None):
    return post("/assistant", json.dumps({"patient": fields, "question": question}, sort_keys=True))


def error_message(body):
    return " ".join(e["msg"] for e in body.get("detail", [])) or "Unexpected response from the API."


def error_type(body):
    detail = body.get("detail") or [{}]
    return detail[0].get("type")


# ---- demo queue ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_queue():
    return json.loads(QUEUE_PATH.read_text(encoding="utf-8"))["patients"]


@st.cache_data(ttl="10m", show_spinner=False)
def score_queue(patients_json):
    """Scores every queue patient through /predict (in parallel; Render can be slow)."""
    patients = json.loads(patients_json)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda p: _post("/predict", p["fields"]), patients))
    return results


# ---- formatting ------------------------------------------------------------

def comparison(score, base_rate):
    """The score relative to the test-set readmission rate, e.g. 'about 1.7× the average readmission rate'."""
    # Always the plain multiple: "below the average" read oddly next to a Moderate tier
    return f"about {score / base_rate:.1f}× the average readmission rate"


def moderate_note(info):
    """Why a Moderate patient can be at or just under 1.0×: the tier starts at the median score."""
    start = info["risk_tiers"]["cutoffs"]["moderate"]
    median = info["risk_tiers"]["percentiles"]["moderate"] == 50
    where = "the median score" if median else f"the {info['risk_tiers']['percentiles']['moderate']}th percentile"
    return (f"The Moderate tier starts at {where} ({start:.1%}), just below the "
            f"{info['performance']['base_rate']:.1%} average readmission rate.")


def tier_scope(tier, percentiles):
    """Where a tier sits among test-set scores, from the configured percentile cutoffs."""
    moderate, high = percentiles["moderate"], percentiles["high"]
    return {
        "High": f"top {100 - high}% of patients",
        "Moderate": f"between the {moderate}th and {high}th percentile of patients",
        "Low": f"lower {moderate}% of patients",
    }[tier]


def tier_badge(tier):
    badge = TIER_BADGE[tier]
    label = tier if tier == "Out of scope" else f"{tier} risk"
    st.badge(label, icon=badge["icon"], color=badge["color"])


def md(text):
    """Escape characters Streamlit markdown would treat specially (e.g. $ for LaTeX)."""
    return str(text).replace("$", "\\$")


def describe_patient(fields, labels):
    age_low, age_high = fields["age"].strip("[)").split("-")
    return {
        "Age group": f"{age_low}-{int(age_high) - 1}",
        "Gender": fields["gender"],
        "Length of stay": f"{fields['time_in_hospital']} days",
        "Prior inpatient stays": fields["number_inpatient"],
        "Primary diagnosis": "Not recorded" if fields["diag_1_group"] == "Missing" else fields["diag_1_group"],
        "Discharge destination": labels.get(str(fields["discharge_disposition_id"]),
                                            str(fields["discharge_disposition_id"])),
    }


# ---- dashboard -------------------------------------------------------------

def render_dashboard(title, source_label, fields, key):
    """The full dashboard for one patient. `key` keeps widget keys unique per patient."""
    info = get("/model-info")
    labels = info["discharge_codes"]["labels"]
    base_rate = info["performance"]["base_rate"]

    st.subheader(title, anchor=False)
    facts = describe_patient(fields, labels)
    st.caption(f"{source_label} · " + " · ".join(f"{k}: {v}" for k, v in facts.items()))

    with st.spinner("Scoring patient..."):
        status, result = explain(fields)
    if status != 200:
        if error_type(result) == "out_of_scope":
            with st.container(border=True):
                tier_badge("Out of scope")
                st.warning(error_message(result), icon=":material/block:")
                st.caption("The assistant is not used for out-of-scope patients.")
        else:
            st.error(error_message(result), icon=":material/error:")
        return

    tier = result["risk_tier"]
    tier_row = next(t for t in info["risk_tiers"]["tiers"] if t["tier"] == tier)

    left, right = st.columns(2)
    with left.container(border=True, height="stretch"):
        st.markdown("**30-day readmission risk**")
        tier_badge(tier)
        st.subheader(f"{tier} risk — {comparison(result['readmission_risk_score'], base_rate)}", anchor=False)
        st.markdown(f"({tier_scope(tier, info['risk_tiers']['percentiles'])})")
        st.caption(f"Score {result['readmission_risk_score']:.0%} (average: {base_rate:.0%})")
        st.caption(f"In the test set, {tier_row['readmission_rate']:.1%} of {tier.lower()}-risk patients "
                   f"were readmitted within 30 days.")
        if tier == "Moderate":
            st.caption(moderate_note(info))

    with right.container(border=True, height="stretch"):
        st.markdown("**What the score is based on**")
        factors = pd.DataFrame([
            {"Factor": f["description"], "Association": ASSOCIATION[f["direction"]], "Share": f["share"] * 100}
            for f in result["factors"][:6]
        ])
        st.dataframe(factors, hide_index=True, column_config={
            "Share": st.column_config.ProgressColumn("Share of explanation", format="%.0f%%",
                                                     min_value=0, max_value=100),
        })
        st.caption("Associations in the model, not causes. Age, gender and race appear because the model "
                   "uses them; they are not things discharge planning can change.")

    with st.container(border=True):
        st.markdown("**Summary**")
        st.write(md(result["summary"]))

    plan, chat = st.columns([3, 2])
    with plan.container(border=True, height="stretch"):
        st.markdown("**Discharge-planning considerations**")
        with st.spinner("Finding sources..."):
            status, reply = ask(fields)
        render_reply(reply if status == 200 else None, error_message(reply) if status != 200 else None)

    with chat.container(border=True, height="stretch"):
        st.markdown("**Ask the assistant**")
        preset = st.pills("Suggested questions", PRESET_QUESTIONS, key=f"preset_{key}")
        with st.form(f"question_form_{key}", border=False):
            typed = st.text_input("Your question", max_chars=500, key=f"question_{key}",
                                  placeholder="Ask about discharge planning for this patient")
            submitted = st.form_submit_button("Ask", icon=":material/send:")
        question = typed.strip() if submitted and typed.strip() else preset
        if question:
            with st.spinner("Checking the sources..."):
                status, answer = ask(fields, question)
            st.markdown(f"**Q:** {md(question)}")
            render_reply(answer if status == 200 else None, error_message(answer) if status != 200 else None)

    render_performance(info)


def render_reply(reply, error=None):
    if error:
        st.error(error, icon=":material/error:")
        return
    if reply["status"] == "refused":
        st.warning(reply["message"], icon=":material/do_not_disturb_on:")
    elif reply["status"] == "insufficient_information":
        st.info(reply["message"], icon=":material/help:")
    else:
        st.caption(md(reply["message"]))
        for kind, heading in (("suggestion", None), ("context", "Background")):
            items = [i for i in reply["items"] if i["type"] == kind]
            if items and heading:
                st.markdown(f"*{heading}*")
            for item in items:
                cites = " ".join(f":gray-badge[{c}]" for c in item["citations"])
                linked = f"  \n:small[Related to: {md(item['linked_factor'])}]" if item["linked_factor"] else ""
                st.markdown(f"- {md(item['text'])} {cites}{linked}")
        if reply["citations"]:
            with st.expander("Supporting evidence", icon=":material/menu_book:"):
                for cid, c in reply["citations"].items():
                    color = "orange" if c["verified_by"] == "pending" else "green"
                    st.markdown(f"**{cid}** · [{md(c['title'])}]({c['url']}) · {md(c['section'])}  \n"
                                f":{color}-badge[{c['review_label']}]")
                    st.markdown(f"> {md(c['quote'])}")
    if reply.get("unsupported"):
        with st.expander("Not shown: unsupported statements", icon=":material/report:"):
            for u in reply["unsupported"]:
                st.markdown(f"- {md(u['text'])}  \n:small[{md(u['reason'])}]")
    if reply.get("mode") == "rule_based_fallback":
        st.caption(md(reply["note"]))
    st.caption(md(reply["disclaimer"]))


def render_performance(info):
    perf = info["performance"]
    high = perf["at_high_cutoff"]
    with st.expander("How well the model performs (test set)", icon=":material/query_stats:"):
        with st.container(horizontal=True):
            low, upper = perf["auc_95_interval"]
            st.metric("AUC", f"{perf['auc']:.3f}", border=True,
                      help=f"95% interval {low:.3f}–{upper:.3f}. 0.5 = chance, 1.0 = perfect ranking.")
            st.metric("Precision (High tier)", f"{high['precision']:.0%}", border=True,
                      help="Share of High-tier patients who were readmitted.")
            st.metric("Recall (High tier)", f"{high['recall']:.0%}", border=True,
                      help="Share of readmitted patients who were in the High tier.")
            st.metric("Specificity (High tier)", f"{high['specificity']:.0%}", border=True,
                      help="Share of patients not readmitted who were outside the High tier.")
        st.caption(f"Computed on {perf['test_encounters']:,} held-out encounters. See *About the model* "
                   "for tiers and subgroup results.")
