"""Builds model_metadata.json from the training data and the saved model.

Every value the API validates against comes from here, so nothing is typed in
by hand. This does not retrain anything: it reproduces the split from
exploration.ipynb, loads readmission_model.pkl, and checks the test AUC still
matches before writing.

Needs clean_diabetic_data.csv (gitignored), so it runs locally:
    python build_metadata.py           # write model_metadata.json
    python build_metadata.py --check   # exit 1 if the committed file is stale
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "clean_diabetic_data.csv"
MODEL_PATH = BASE_DIR / "readmission_model.pkl"
FEATURES_PATH = BASE_DIR / "model_features.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"
# ID descriptions shipped with the UCI dataset
IDS_MAPPING_PATH = BASE_DIR / "reference" / "IDS_mapping.csv"
NOT_RECORDED = {"NULL", "Not Mapped"}

# Test AUC of the saved model; a mismatch means the model or the preprocessing changed
EXPECTED_TEST_AUC = 0.6642
MIN_CATEGORY_ROWS = 50
OUT_OF_SCOPE_DISCHARGE_CODES = [11, 13, 14, 19, 20, 21]  # expired / hospice

NUMERIC_FIELDS = ["number_inpatient", "number_emergency", "number_diagnoses",
                  "num_medications", "time_in_hospital", "num_lab_procedures"]
CATEGORICAL_FIELDS = ["diabetesMed", "diag_1_group", "diag_2_group", "diag_3_group",
                      "age", "gender", "race"]


def categorize_diagnosis(code):
    if pd.isna(code):
        return "Missing"
    if code.startswith("V") or code.startswith("E"):
        return "Other"

    code_num = float(code)

    if 250 <= code_num < 251:
        return "Diabetes"
    elif 390 <= code_num <= 459 or code_num == 785:
        return "Circulatory"
    elif 460 <= code_num <= 519 or code_num == 786:
        return "Respiratory"
    elif 520 <= code_num <= 579 or code_num == 787:
        return "Digestive"
    elif 800 <= code_num <= 999:
        return "Injury"
    elif 710 <= code_num <= 739:
        return "Musculoskeletal"
    elif 580 <= code_num <= 629 or code_num == 788:
        return "Genitourinary"
    elif 140 <= code_num <= 239:
        return "Neoplasms"
    else:
        return "Other"


def load_discharge_labels():
    """Reads the discharge_disposition_id section of IDS_mapping.csv."""
    labels, in_section = {}, False
    with open(IDS_MAPPING_PATH, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and row[0] == "discharge_disposition_id":
                in_section = True
            elif in_section:
                if not row or not row[0].strip():
                    break
                description = row[1].strip().rstrip(" .")
                labels[int(row[0])] = "Not recorded" if description in NOT_RECORDED else description
    return labels


def load_split():
    """Same steps as exploration.ipynb cells 0-25 (column order matters for the model)."""
    df = pd.read_csv(CSV_PATH)
    df["max_glu_serum"] = df["max_glu_serum"].fillna("None")
    df["A1Cresult"] = df["A1Cresult"].fillna("None")
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
    df["diag_1"] = df["diag_1"].astype(str)
    df["diag_2"] = df["diag_2"].astype(str)
    df["diag_1_group"] = df["diag_1"].apply(categorize_diagnosis)
    df["diag_3"] = df["diag_3"].replace("?", pd.NA)
    df["diag_3_group"] = df["diag_3"].apply(categorize_diagnosis)
    df["diag_2_group"] = df["diag_2"].apply(categorize_diagnosis)
    df = df.drop(columns=["diag_1", "diag_2", "diag_3", "readmitted"])
    df = df[~df["discharge_disposition_id"].isin(OUT_OF_SCOPE_DISCHARGE_CODES)].reset_index(drop=True)

    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    X = df_encoded.drop(columns=["readmitted_30d", "encounter_id", "patient_nbr"])
    y = df_encoded["readmitted_30d"]

    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=df_encoded["patient_nbr"]))
    return df, X, y, train_idx, test_idx


def build():
    df, X, y, train_idx, test_idx = load_split()
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)

    proba = model.predict_proba(X.iloc[test_idx][features])[:, 1]
    test_auc = round(float(roc_auc_score(y.iloc[test_idx], proba)), 4)
    if test_auc != EXPECTED_TEST_AUC:
        raise SystemExit(f"Test AUC {test_auc} != expected {EXPECTED_TEST_AUC}; "
                         "the model or preprocessing changed.")

    train = df.iloc[train_idx]

    numeric_ranges = {f: {"min": int(train[f].min()), "max": int(train[f].max())}
                      for f in NUMERIC_FIELDS}

    categories, excluded = {}, {}
    for f in CATEGORICAL_FIELDS:
        counts = train[f].value_counts()
        categories[f] = sorted(v for v, n in counts.items() if n >= MIN_CATEGORY_ROWS)
        rare = {v: int(n) for v, n in counts.items() if n < MIN_CATEGORY_ROWS}
        if rare:
            excluded[f] = dict(sorted(rare.items()))

    code_counts = train["discharge_disposition_id"].value_counts()
    allowed_codes = sorted(int(c) for c, n in code_counts.items() if n >= MIN_CATEGORY_ROWS)
    labels = load_discharge_labels()
    discharge_codes = {
        "allowed": allowed_codes,
        "out_of_scope": OUT_OF_SCOPE_DISCHARGE_CODES,
        "excluded_rare": {str(c): int(n) for c, n in sorted(code_counts.items())
                          if n < MIN_CATEGORY_ROWS},
        "labels": {str(c): labels[c] for c in sorted(allowed_codes + OUT_OF_SCOPE_DISCHARGE_CODES)},
    }

    return {
        "model_version": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()[:12],
        "model_features": len(features),
        "split": {"train_rows": len(train_idx), "test_rows": len(test_idx)},
        "test_auc": test_auc,
        "min_category_rows": MIN_CATEGORY_ROWS,
        "numeric_ranges": numeric_ranges,
        "categories": categories,
        "excluded_rare_categories": excluded,
        "discharge_codes": discharge_codes,
    }


def to_json(metadata):
    return json.dumps(metadata, indent=2) + "\n"


if __name__ == "__main__":
    text = to_json(build())
    if "--check" in sys.argv:
        current = METADATA_PATH.read_text(encoding="utf-8") if METADATA_PATH.exists() else ""
        if current != text:
            raise SystemExit("model_metadata.json is out of date; run python build_metadata.py")
        print("model_metadata.json is up to date")
    else:
        METADATA_PATH.write_text(text, encoding="utf-8")
        print(f"wrote {METADATA_PATH.name}")
