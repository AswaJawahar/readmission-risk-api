"""Turns SHAP contributions into plain-language factors, one per API input field.

SHAP gives one contribution per encoded model column (51). Columns created by
one-hot encoding a field are summed back into that field, so every factor shown
is one of the 14 inputs the user entered.
"""
NUMERIC_FIELDS = ["number_inpatient", "discharge_disposition_id", "number_emergency",
                  "number_diagnoses", "num_medications", "time_in_hospital", "num_lab_procedures"]
CATEGORICAL_FIELDS = ["diabetesMed", "diag_1_group", "diag_2_group", "diag_3_group",
                      "age", "gender", "race"]
DEMOGRAPHIC_FIELDS = {"age", "gender", "race"}

# Plain-language names, following the dataset's field definitions (Strack et al., 2014)
LABELS = {
    "number_inpatient": "Inpatient stays in the year before this admission",
    "number_emergency": "Emergency visits in the year before this admission",
    "discharge_disposition_id": "Discharge destination",
    "number_diagnoses": "Diagnoses recorded",
    "num_medications": "Different medications given during this stay",
    "time_in_hospital": "Length of stay",
    "num_lab_procedures": "Lab tests during this stay",
    "diabetesMed": "Diabetes medication prescribed",
    "diag_1_group": "Primary diagnosis",
    "diag_2_group": "Secondary diagnosis",
    "diag_3_group": "Additional diagnosis",
    "age": "Age group",
    "gender": "Gender",
    "race": "Race",
}

# Below this share of the total contribution a factor is described as having little effect
MINIMAL_SHARE = 0.02
# Factors below this share are left out of the one-sentence summary
SUMMARY_MIN_SHARE = 0.05


def field_columns(field, features):
    if field in NUMERIC_FIELDS:
        return [field]
    return [c for c in features if c.startswith(field + "_")]


def describe(field, value, discharge_labels):
    """The patient's value for a field, written for discharge staff."""
    if field == "number_inpatient":
        return f"{value} inpatient stay{'s' * (value != 1)} in the year before this admission"
    if field == "number_emergency":
        return f"{value} emergency visit{'s' * (value != 1)} in the year before this admission"
    if field == "discharge_disposition_id":
        return f"Discharge destination: {discharge_labels[str(value)]}"
    if field == "number_diagnoses":
        return f"{value} diagnos{'is' if value == 1 else 'es'} recorded"
    if field == "num_medications":
        return f"{value} different medication{'s' * (value != 1)} given during this stay"
    if field == "time_in_hospital":
        return f"{value}-day hospital stay"
    if field == "num_lab_procedures":
        return f"{value} lab test{'s' * (value != 1)} during this stay"
    if field == "diabetesMed":
        return "Diabetes medication prescribed" if value == "Yes" else "No diabetes medication prescribed"
    if field.startswith("diag_"):
        shown = "not recorded" if value == "Missing" else value
        return f"{LABELS[field]}: {shown}"
    if field == "age":
        low, high = value.strip("[)").split("-")
        return f"Age group: {low}-{int(high) - 1}"
    return f"{LABELS[field]}: {value}"


def explain(patient, shap_row, features, discharge_labels):
    """Factors for one patient, largest effect first.

    `patient` is the validated input dict and `shap_row` the SHAP values (log-odds)
    for its encoded row, in the same order as `features`.
    """
    contributions = dict(zip(features, shap_row))
    factors = []
    for field in NUMERIC_FIELDS + CATEGORICAL_FIELDS:
        contribution = float(sum(contributions[c] for c in field_columns(field, features)))
        factors.append({"field": field, "label": LABELS[field], "value": patient[field],
                        "description": describe(field, patient[field], discharge_labels),
                        "contribution_log_odds": contribution,
                        "demographic": field in DEMOGRAPHIC_FIELDS})

    total = sum(abs(f["contribution_log_odds"]) for f in factors)
    for f in factors:
        f["share"] = abs(f["contribution_log_odds"]) / total if total else 0.0
        if f["share"] < MINIMAL_SHARE:
            f["direction"] = "minimal"
        else:
            f["direction"] = "raises" if f["contribution_log_odds"] > 0 else "lowers"
    factors.sort(key=lambda f: abs(f["contribution_log_odds"]), reverse=True)

    for f in factors:
        f["contribution_log_odds"] = round(f["contribution_log_odds"], 4)
        f["share"] = round(f["share"], 4)
    return factors


def summarize(factors, top_n=2):
    """A one-sentence, template-based summary of the main factors (no AI involved)."""
    def listed(items, verb):
        names = [f["description"][0].lower() + f["description"][1:] for f in items]
        if len(names) == 1:
            return f"factor {verb} the most is {names[0]}"
        return f"factors {verb} the most are " + ", ".join(names[:-1]) + " and " + names[-1]

    notable = [f for f in factors if f["share"] >= SUMMARY_MIN_SHARE]
    raising = [f for f in notable if f["direction"] == "raises"][:top_n]
    lowering = [f for f in notable if f["direction"] == "lowers"][:top_n]
    if not raising and not lowering:
        return "No single factor stands out; the score is close to the model's typical patient."

    parts = [(raising, "raising this patient's risk"), (lowering, "lowering this patient's risk")]
    # Lead with whichever direction carries more of the explanation
    parts.sort(key=lambda part: sum(f["share"] for f in part[0]), reverse=True)
    phrases = ["the " + listed(items, verb) for items, verb in parts if items]
    if len(phrases) == 2:
        phrases[1] = phrases[1].replace("this patient's risk", "it")
    sentence = "; ".join(phrases) + "."
    return sentence[0].upper() + sentence[1:]
