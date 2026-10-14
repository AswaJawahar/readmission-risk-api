"""Builds demo/patient_queue.json: the demo patient queue for the Streamlit app.

24 encounters from the test set (8 per risk tier, chosen with a fixed seed and
varied diagnoses and discharge destinations), plus one hand-made hospice case
that shows the out-of-scope check. Only the 14 API fields are kept; encounter
and patient numbers are dropped and replaced with fake IDs.

Needs clean_diabetic_data.csv (gitignored), so it runs locally:
    python build_patient_queue.py           # write demo/patient_queue.json
    python build_patient_queue.py --check   # exit 1 if the committed file is stale
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

from build_metadata import BASE_DIR, CSV_PATH, FEATURES_PATH, METADATA_PATH, MODEL_PATH, load_split
from explain import CATEGORICAL_FIELDS, NUMERIC_FIELDS
from risk import TIERS, assign_tier

QUEUE_PATH = BASE_DIR / "demo" / "patient_queue.json"
PER_TIER = 8
SEED = 2026

HOSPICE_CASE = {
    "number_inpatient": 2, "discharge_disposition_id": 14, "number_emergency": 1,
    "number_diagnoses": 9, "num_medications": 18, "time_in_hospital": 6,
    "num_lab_procedures": 52, "diabetesMed": "Yes", "diag_1_group": "Neoplasms",
    "diag_2_group": "Circulatory", "diag_3_group": "Diabetes", "age": "[80-90)",
    "gender": "Male", "race": "Caucasian",
}


def accepted(row, metadata):
    """True if every value is one the API accepts."""
    if row["discharge_disposition_id"] not in metadata["discharge_codes"]["allowed"]:
        return False
    for field, bounds in metadata["numeric_ranges"].items():
        if not bounds["min"] <= row[field] <= bounds["max"]:
            return False
    return all(row[f] in metadata["categories"][f] for f in CATEGORICAL_FIELDS)


def pick_varied(candidates, n, rng):
    """Random order, preferring rows whose (primary diagnosis, discharge code) pair is new."""
    order = rng.permutation(len(candidates))
    chosen, seen = [], set()
    for strict in (True, False):
        for i in order:
            row = candidates[i]
            pair = (row["diag_1_group"], row["discharge_disposition_id"])
            if i in chosen or (strict and pair in seen):
                continue
            chosen.append(i)
            seen.add(pair)
            if len(chosen) == n:
                return [candidates[j] for j in chosen]
    return [candidates[j] for j in chosen]


def build():
    df, X, _, _, test_idx = load_split()
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    cutoffs = metadata["risk_tiers"]["cutoffs"]

    test = df.iloc[test_idx].reset_index(drop=True)
    scores = model.predict_proba(X.iloc[test_idx][features])[:, 1]
    fields = NUMERIC_FIELDS + CATEGORICAL_FIELDS
    rows = [{f: (v.item() if hasattr(v, "item") else v) for f, v in zip(fields, values)}
            for values in test[fields].itertuples(index=False)]

    rng = np.random.default_rng(SEED)
    top = int(np.argmax(scores))
    chosen = [rows[top]]  # the test set's highest-scoring encounter, used throughout the tests
    used = {top}
    for tier in TIERS:
        pool = [rows[i] for i, s in enumerate(scores)
                if i not in used and assign_tier(s, cutoffs) == tier and accepted(rows[i], metadata)]
        chosen += pick_varied(pool, PER_TIER - (tier == "High"), rng)

    patients = [{"source": "test_set", "fields": row} for row in chosen]
    patients.append({"source": "hand_made",
                     "note": "Created by hand to show the out-of-scope check; not a dataset record.",
                     "fields": HOSPICE_CASE})
    order = rng.permutation(len(patients))
    ids = rng.choice(np.arange(1000, 10000), size=len(patients), replace=False)
    queue = [{"id": f"PT-{ids[k]}", **patients[i]} for k, i in enumerate(order)]

    return {
        "description": ("Demo patient queue. Test-set encounters are de-identified records from the UCI "
                        "Diabetes 130-US Hospitals dataset (14 model fields only, fake IDs); the "
                        "hand-made case is invented."),
        "generated_by": "build_patient_queue.py",
        "patients": queue,
    }


def to_json(queue):
    return json.dumps(queue, indent=2) + "\n"


if __name__ == "__main__":
    text = to_json(build())
    if "--check" in sys.argv:
        current = QUEUE_PATH.read_text(encoding="utf-8") if QUEUE_PATH.exists() else ""
        if current != text:
            raise SystemExit("demo/patient_queue.json is out of date; run python build_patient_queue.py")
        print("demo/patient_queue.json is up to date")
    else:
        QUEUE_PATH.parent.mkdir(exist_ok=True)
        QUEUE_PATH.write_text(text, encoding="utf-8")
        print(f"wrote {QUEUE_PATH.relative_to(BASE_DIR)}")
