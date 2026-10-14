import pandas as pd
import streamlit as st

from ui import get

st.title("About the model")
info = get("/model-info")
perf = info["performance"]
high = perf["at_high_cutoff"]
low, upper = perf["auc_95_interval"]

st.markdown(
    "A Gradient Boosting model trained on 79,541 encounters from the UCI *Diabetes 130-US Hospitals* "
    "dataset (1999–2008), using 14 fields. All numbers below were computed on "
    f"{perf['test_encounters']:,} held-out test encounters, with each patient's records kept entirely "
    "in either the training or the test set."
)

with st.container(horizontal=True):
    st.metric("AUC", f"{perf['auc']:.3f}", border=True, help=f"95% interval {low:.3f}–{upper:.3f}")
    st.metric("30-day readmission rate", f"{perf['base_rate']:.1%}", border=True,
              help="The 'average' used for comparisons on each patient's dashboard.")
    st.metric("Precision (High tier)", f"{high['precision']:.0%}", border=True)
    st.metric("Recall (High tier)", f"{high['recall']:.0%}", border=True)
    st.metric("Specificity (High tier)", f"{high['specificity']:.0%}", border=True)

st.subheader("Risk tiers")
percentiles = info["risk_tiers"]["percentiles"]
st.caption(f"Tiers are cut at the {percentiles['moderate']}th and {percentiles['high']}th percentiles of "
           "test-set scores. The tier rates come from the same test set, so they describe it rather than "
           "an independent validation.")
tiers = pd.DataFrame([{
    "Tier": t["tier"],
    "Score range": f"{t['min_score']:.1%} – {t['max_score']:.1%}",
    "Test encounters": t["encounters"],
    "Readmitted within 30 days": t["readmission_rate"] * 100,
} for t in info["risk_tiers"]["tiers"]])
st.dataframe(tiers, hide_index=True, column_config={
    "Test encounters": st.column_config.NumberColumn(format="localized"),
    "Readmitted within 30 days": st.column_config.NumberColumn(format="%.1f%%"),
})

st.subheader("Results by group")
st.caption("Groups with fewer than 100 readmissions in the test set show counts only. "
           "The model ranks patients aged 70+ less well than younger patients.")
groups = pd.DataFrame([{
    "Group": f"{g['group']}: {g['value']}",
    "Test encounters": g["encounters"],
    "Readmitted": g["readmission_rate"] * 100,
    "AUC": g.get("auc"),
    "AUC 95% interval": (f"{g['auc_95_interval'][0]:.3f}–{g['auc_95_interval'][1]:.3f}"
                         if "auc_95_interval" in g else "Too few readmissions"),
    "Readmissions caught by High tier": (g["recall_at_high_cutoff"] * 100 if "recall_at_high_cutoff" in g
                                         else None),
} for g in info["subgroups"]])
st.dataframe(groups, hide_index=True, placeholder="—", height=35 * (len(groups) + 1) + 3, column_config={
    "Test encounters": st.column_config.NumberColumn(format="localized"),
    "Readmitted": st.column_config.NumberColumn(format="%.1f%%"),
    "AUC": st.column_config.NumberColumn(format="%.3f"),
    "Readmissions caught by High tier": st.column_config.NumberColumn(format="%.1f%%"),
})

st.subheader("Knowledge base")
config = get("/assistant/config")
sources = pd.DataFrame([{
    "Source": s["title"], "Publisher": s["publisher"], "Year": s["year"], "Link": s["url"],
    "Excerpts": s["excerpts"], "Pending review": s["pending_review"],
} for s in get("/sources")])
st.caption("The assistant only uses these curated sources. "
           f"Mode: {'LLM' if config['active_mode'] == 'llm' else 'rule-based (no LLM)'}. "
           + ("Only reviewer-approved excerpts are used." if config["kb_require_verified"]
              else "Excerpts not yet reviewed are used and labelled 'Source pending review'."))
st.dataframe(sources, hide_index=True, column_config={
    "Link": st.column_config.LinkColumn(display_text="Open"),
    "Year": st.column_config.NumberColumn(format="%d"),
})

st.subheader("Limitations")
st.markdown(
    "- Trained on 1999–2008 administrative data; may not reflect current practice.\n"
    "- Uses 14 fields only (no lab results, individual medications or social factors).\n"
    "- An AUC around 0.66 means the model ranks patients only moderately well.\n"
    "- The assistant's scope checks are keyword-based and not robust against determined attempts.\n"
    "- Advisory only: not a diagnosis or treatment recommendation, and not validated for clinical use."
)
