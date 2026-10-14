"""Curated knowledge base (knowledge_base/sources.json) and retrieval over it.

Retrieval is TF-IDF text similarity plus a boost for excerpts tagged with the
topics of the patient's risk-raising factors. Results below MIN_SCORE are
dropped, so an unrelated question returns nothing rather than a weak match.
"""
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from explain import LABELS

KB_PATH = Path(__file__).resolve().parent / "knowledge_base" / "sources.json"

TAGS = {
    "prior_admissions", "emergency_visits", "discharge_destination", "care_transitions",
    "discharge_communication", "medications", "medication_reconciliation",
    "diabetes_medications", "self_management", "patient_education", "follow_up",
    "length_of_stay", "comorbidities", "primary_diagnosis", "hba1c", "hypoglycemia",
    "readmission_context", "age",
}

# Knowledge-base topics for each model input. Gender and race are left out on
# purpose (no discharge guidance should be chosen because of them), and so is
# the lab-test count, which no source addresses.
FIELD_TAGS = {
    "number_inpatient": ["prior_admissions", "readmission_context"],
    "number_emergency": ["emergency_visits"],
    "discharge_disposition_id": ["discharge_destination", "care_transitions"],
    "num_medications": ["medications", "medication_reconciliation"],
    "diabetesMed": ["diabetes_medications", "self_management"],
    "time_in_hospital": ["length_of_stay"],
    "number_diagnoses": ["comorbidities"],
    "diag_1_group": ["primary_diagnosis"],
    "diag_2_group": ["comorbidities"],
    "diag_3_group": ["comorbidities"],
    "age": ["age"],
}

TAG_WEIGHT = 0.3
# Unrelated questions score 0; relevant free-text questions score about 0.09 and up
MIN_SCORE = 0.08
# AHRQ writes these as one word; split them so they match "follow-up" etc.
SPELLINGS = {"followup": "follow up", "postdischarge": "post discharge", "posthospital": "post hospital"}


def normalize(text):
    text = text.lower()
    for joined, split in SPELLINGS.items():
        text = text.replace(joined, split)
    return text


def citation(source):
    return {k: source[k] for k in ("source_id", "title", "publisher", "year", "url")}


class KnowledgeBase:
    def __init__(self, path=KB_PATH):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.sources = {s["source_id"]: s for s in data["sources"]}
        self.excerpts = [dict(e, source_id=s["source_id"])
                         for s in data["sources"] for e in s["excerpts"]]
        documents = [" ".join([e["summary"], e["quote"], e["section"]]
                              + [t.replace("_", " ") for t in e["tags"]])
                     for e in self.excerpts]
        self.vectorizer = TfidfVectorizer(preprocessor=normalize, stop_words="english",
                                          ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(documents)

    def search(self, query="", tags=(), top_k=5, require_verified=False):
        """Excerpts matching a question and/or topic tags, best first.

        With require_verified=True, excerpts still marked "pending" are skipped.
        """
        tags = set(tags)
        text = " ".join([query] + [t.replace("_", " ") for t in sorted(tags)]).strip()
        if not text:
            return []
        similarity = linear_kernel(self.vectorizer.transform([text]), self.matrix)[0]

        results = []
        for excerpt, sim in zip(self.excerpts, similarity):
            if require_verified and excerpt["verified_by"] == "pending":
                continue
            overlap = len(tags & set(excerpt["tags"])) / len(tags) if tags else 0.0
            score = float(sim) + TAG_WEIGHT * overlap
            if score >= MIN_SCORE:
                results.append({**excerpt, "score": round(score, 4),
                                "source": citation(self.sources[excerpt["source_id"]])})
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def search_for_factor(self, field, top_k=3, require_verified=False):
        """Excerpts related to one model input; empty for fields with no topics."""
        if field not in FIELD_TAGS:
            return []
        return self.search(LABELS[field], FIELD_TAGS[field], top_k, require_verified)

    def source_list(self):
        return [{**citation(s), "accessed": s["accessed"], "excerpts": len(s["excerpts"]),
                 "pending_review": sum(e["verified_by"] == "pending" for e in s["excerpts"])}
                for s in self.sources.values()]
