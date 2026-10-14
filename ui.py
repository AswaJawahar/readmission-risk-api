"""Shared pieces of the Streamlit app: API client, formatting and the patient dashboard.

Every number shown comes from the API (the live model and model_metadata.json).
"""
import html
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_API_URL = "https://readmission-risk-api-oyvy.onrender.com"
QUEUE_PATH = BASE_DIR / "demo" / "patient_queue.json"
CSS_PATH = BASE_DIR / "assets" / "style.css"

TIER_CLASS = {"High": "high", "Moderate": "moderate", "Low": "low", "Out of scope": "oos"}
# CSS variable prefix for each tier in style.css
TIER_VAR = {"High": "high", "Moderate": "mod", "Low": "low", "Out of scope": "oos"}
# Short, plain labels for long discharge destinations. Display only: the API and the
# model keep the dataset's own labels (reference/IDS_mapping.csv).
SHORT_DISCHARGE = {
    1: "Home", 2: "Another short-term hospital", 3: "Skilled nursing facility",
    4: "Intermediate care facility", 5: "Other inpatient care", 6: "Home with home health",
    7: "Left against medical advice", 8: "Home with IV provider", 11: "Expired",
    13: "Hospice (home)", 14: "Hospice (medical facility)", 15: "Swing bed (same hospital)",
    18: "Not recorded (code 18)", 19: "Expired at home (hospice)", 20: "Expired in a facility (hospice)",
    21: "Expired, place unknown (hospice)", 22: "Rehab facility", 23: "Long-term care hospital",
    25: "Not recorded (code 25)", 28: "Psychiatric hospital",
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

def setting(name):
    """A setting from the environment or Streamlit secrets (None if unset)."""
    value = os.environ.get(name)
    if not value:
        try:
            value = st.secrets.get(name)
        except Exception:  # no secrets file
            value = None
    return value


def api_url():
    """API_URL from the environment or Streamlit secrets, else the deployed API."""
    return (setting("API_URL") or DEFAULT_API_URL).rstrip("/")


# Mock sign-in: each role's demo key (only needed when the API has ROLE_KEYS configured)
ROLE_KEY_SETTINGS = {"discharge_staff": "STAFF_API_KEY", "auditor": "AUDITOR_API_KEY"}


def role_key(role):
    return setting(ROLE_KEY_SETTINGS[role])


def _headers(key):
    return {"X-API-Key": key} if key else {}


def _post(path, payload, key=None):
    try:
        response = requests.post(f"{api_url()}{path}", json=payload, headers=_headers(key), timeout=90)
    except requests.RequestException as exc:
        return 0, {"detail": [{"type": "connection", "msg": f"Could not reach the API: {exc}"}]}
    return response.status_code, response.json()


def _get(path, key=None):
    response = requests.get(f"{api_url()}{path}", headers=_headers(key), timeout=90)
    return response.status_code, response.json()


@st.cache_data(ttl="10m", max_entries=200, show_spinner=False)
def post(path, payload_json, key=None):
    """Cached POST; the payload is passed as JSON text so it hashes reliably."""
    return _post(path, json.loads(payload_json), key)


@st.cache_data(ttl="1h", show_spinner=False)
def get(path):
    status, body = _get(path)
    if status != 200:
        raise RuntimeError(f"GET {path} failed with HTTP {status}")
    return body


def audit_entries(limit=500):
    """Not cached: the auditor should always see the latest entries."""
    return _get(f"/audit?limit={limit}", role_key("auditor"))


def explain(fields):
    return post("/explain", json.dumps(fields, sort_keys=True), role_key("discharge_staff"))


def ask(fields, question=None):
    return post("/assistant", json.dumps({"patient": fields, "question": question}, sort_keys=True),
                role_key("discharge_staff"))


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
    key = role_key("discharge_staff")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda p: _post("/predict", p["fields"], key), patients))
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


# ---- styling ---------------------------------------------------------------

def theme_class():
    """'cp-dark' or 'cp-light', matching the viewer's active Streamlit theme."""
    try:
        return "cp-dark" if st.context.theme.type == "dark" else "cp-light"
    except Exception:  # e.g. no browser context in tests
        return "cp-light"


@st.cache_data(show_spinner=False)
def _css():
    return CSS_PATH.read_text(encoding="utf-8")


def inject_css():
    st.html(f"<style>{_css()}</style>")


def css_vars(theme):
    """The --cp-* colour variables defined for `theme` ('cp-light' or 'cp-dark') in style.css."""
    block = re.search(r"\." + theme + r"\s*\{(.*?)\}", _css(), re.S).group(1)
    return dict(re.findall(r"--cp-([\w-]+):\s*([^;]+);", block))


def esc(value):
    return html.escape(str(value))


def themed(body):
    st.html(f'<div class="{theme_class()}">{body}</div>')


def discharge_short(code, labels):
    return SHORT_DISCHARGE.get(int(code), labels.get(str(code), str(code)))


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
        "Discharge destination": discharge_short(fields["discharge_disposition_id"], labels),
    }


def patient_chips(source_label, facts):
    chips = [f'<span class="cp-chip">{esc(source_label)}</span>']
    chips += [f'<span class="cp-chip"><b>{esc(k)}:</b> {esc(v)}</span>' for k, v in facts.items()]
    themed(f'<div class="cp-chips">{"".join(chips)}</div>')


def risk_card(result, info, tier_row):
    """Tier-coloured card: badge, large multiple, percentile bar, then the smaller details."""
    tier, score = result["risk_tier"], result["readmission_risk_score"]
    base_rate = info["performance"]["base_rate"]
    pct = info["risk_tiers"]["percentiles"]
    widths = {"low": pct["moderate"], "moderate": pct["high"] - pct["moderate"], "high": 100 - pct["high"]}
    segments = "".join(f'<div class="cp-pbar-seg {k}" style="width:{w}%"></div>' for k, w in widths.items())
    seg_labels = "".join(f'<span style="width:{w}%">{k.capitalize()}</span>' for k, w in widths.items())
    marker_at = min(max(result["risk_percentile"], 1), 99)
    details = [f"Score {score:.0%} (average: {base_rate:.0%})",
               (f"Among the top 1% of test-set scores" if result["risk_percentile"] >= 99
                else f"Higher than about {result['risk_percentile']}% of test-set patients"),
               f"In the test set, {tier_row['readmission_rate']:.1%} of {tier.lower()}-risk patients "
               f"were readmitted within 30 days."]
    if tier == "Moderate":
        details.append(moderate_note(info))
    sentence = f"{tier} risk — {comparison(score, base_rate)}"
    scope = tier_scope(tier, pct)
    themed(f"""
<div class="cp-risk cp-tier-{TIER_CLASS[tier]}">
  <div class="cp-risk-title">30-day readmission risk</div>
  <span class="cp-badge">{esc(tier)} risk</span>
  <div class="cp-multiple" title="{esc(sentence)}">
    <span class="cp-multiple-value">{score / base_rate:.1f}×</span>
    <span class="cp-multiple-label">the average readmission rate</span>
  </div>
  <div class="cp-scope">{esc(scope[0].upper() + scope[1:])}</div>
  <div class="cp-pbar" role="img" aria-label="Percentile {result['risk_percentile']} of test-set scores">
    {segments}<div class="cp-pbar-marker" style="left:{marker_at}%"></div>
  </div>
  <div class="cp-pbar-labels">{seg_labels}</div>
  <div class="cp-risk-foot">{"<br>".join(esc(d) for d in details)}</div>
</div>""")


def out_of_scope_card(message):
    themed(f"""
<div class="cp-risk cp-tier-oos">
  <div class="cp-risk-title">30-day readmission risk</div>
  <span class="cp-badge">Out of scope</span>
  <div class="cp-risk-headline">No prediction is made for this patient</div>
  <div class="cp-risk-foot">{esc(message)}<br>The assistant is not used for out-of-scope patients.</div>
</div>""")


def factor_bars(factors, labels):
    rows = []
    for f in factors[:6]:
        description = f["description"]
        if f["field"] == "discharge_disposition_id":
            description = f"Discharge destination: {discharge_short(f['value'], labels)}"
        rows.append(f"""
<div class="cp-factor cp-dir-{f['direction']}">
  <div><div class="cp-factor-name">{esc(description)}</div>
       <div class="cp-factor-bar"><div class="cp-factor-fill" style="width:{f['share'] * 100:.0f}%"></div></div></div>
  <div class="cp-factor-dir">{ASSOCIATION[f['direction']]}</div>
  <div class="cp-factor-share">{f['share']:.0%}</div>
</div>""")
    themed(f"""<div class="cp-factors">{"".join(rows)}
<div class="cp-factor-note">Bars show each factor's share of the explanation. These are associations in the
model, not causes. Age, gender and race appear because the model uses them; they are not things discharge
planning can change.</div></div>""")


def tier_cell_style(tier):
    """Background and text colour for a queue tier cell, from style.css for the active theme."""
    var = TIER_VAR.get(tier)
    if var is None:
        return ""
    colors = css_vars(theme_class())
    return f"background-color: {colors[var + '-bg']}; color: {colors[var + '-text']}; font-weight: 600"


# ---- dashboard -------------------------------------------------------------

def render_dashboard(title, source_label, fields, key):
    """The full dashboard for one patient. `key` keeps widget keys unique per patient."""
    info = get("/model-info")
    labels = info["discharge_codes"]["labels"]

    st.subheader(title, anchor=False)
    patient_chips(source_label, describe_patient(fields, labels))

    with st.spinner("Scoring patient..."):
        status, result = explain(fields)
    if status != 200:
        if error_type(result) == "out_of_scope":
            out_of_scope_card(error_message(result))
        else:
            st.error(error_message(result), icon=":material/error:")
        return

    tier = result["risk_tier"]
    tier_row = next(t for t in info["risk_tiers"]["tiers"] if t["tier"] == tier)

    left, right = st.columns(2)
    with left:
        risk_card(result, info, tier_row)

    with right.container(border=True, height="stretch"):
        st.markdown("**What the score is based on**")
        factor_bars(result["factors"], labels)

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
