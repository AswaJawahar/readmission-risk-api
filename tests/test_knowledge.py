import json
import re
from datetime import date

import pytest
from fastapi.testclient import TestClient

from explain import LABELS
from knowledge import FIELD_TAGS, KB_PATH, TAGS, KnowledgeBase
from main import app

client = TestClient(app)
kb = KnowledgeBase()
data = json.loads(KB_PATH.read_text(encoding="utf-8"))
excerpts = [e for s in data["sources"] for e in s["excerpts"]]


def test_sources_have_required_fields():
    for source in data["sources"]:
        for key in ("source_id", "title", "publisher", "year", "url", "retrieved_from", "accessed"):
            assert source[key], f"{source['source_id']} is missing {key}"
        assert source["url"].startswith("https://") and source["retrieved_from"].startswith("https://")
        date.fromisoformat(source["accessed"])


def test_ids_are_unique_and_prefixed_by_source():
    source_ids = [s["source_id"] for s in data["sources"]]
    assert len(source_ids) == len(set(source_ids))
    excerpt_ids = [e["excerpt_id"] for e in excerpts]
    assert len(excerpt_ids) == len(set(excerpt_ids))
    for source in data["sources"]:
        for e in source["excerpts"]:
            assert re.fullmatch(re.escape(source["source_id"]) + r"-\d{2}", e["excerpt_id"])


def test_excerpts_are_short_tagged_and_reviewable():
    for e in excerpts:
        assert e["kind"] in {"recommendation", "evidence", "background", "limitation"}
        assert e["quote"] and e["summary"] and e["section"]
        # Brief quotes only; the summary carries the meaning
        assert len(e["quote"].split()) <= 60
        assert e["tags"] and set(e["tags"]) <= TAGS
        assert isinstance(e["verified_by"], str) and e["verified_by"].strip()


def test_every_excerpt_awaits_review():
    # Flip to a per-reviewer check once the team starts signing entries off
    assert {e["verified_by"] for e in excerpts} == {"pending"}


def test_every_factor_topic_is_covered():
    used = {t for e in excerpts for t in e["tags"]}
    for field, tags in FIELD_TAGS.items():
        assert set(tags) <= used, f"no excerpt covers {field}"


def test_demographics_other_than_age_never_drive_retrieval():
    assert "gender" not in FIELD_TAGS and "race" not in FIELD_TAGS
    assert kb.search_for_factor("race") == [] and kb.search_for_factor("gender") == []


@pytest.mark.parametrize("field,expected_tag", [
    ("number_inpatient", "prior_admissions"),
    ("num_medications", "medication_reconciliation"),
    ("discharge_disposition_id", "discharge_destination"),
    ("number_emergency", "emergency_visits"),
])
def test_factor_retrieval_finds_matching_topics(field, expected_tag):
    results = kb.search_for_factor(field)
    assert results and expected_tag in results[0]["tags"]
    assert results[0]["source"]["url"].startswith("https://")


@pytest.mark.parametrize("question,expected_ids", [
    ("What should we do about medications before discharge?",
     {"ADA-SOC-2026-S16-06", "ADA-SOC-2026-S16-10", "PSNET-MEDREC-04"}),
    ("What education should the patient get?",
     {"AHRQ-RED-TOOL1-05", "AHRQ-RED-TOOL1-06", "AHRQ-RED-TOOL1-07"}),
    ("Why does a patient with many prior admissions need extra attention?",
     {"ADA-SOC-2026-S16-12"}),
])
def test_questions_retrieve_relevant_excerpts(question, expected_ids):
    top_ids = {r["excerpt_id"] for r in kb.search(question, top_k=3)}
    assert expected_ids & top_ids


@pytest.mark.parametrize("question", ["What is the capital of France?", "Tell me a joke",
                                      "Who won the football game", ""])
def test_unrelated_questions_return_nothing(question):
    assert kb.search(question) == []


def test_results_are_ranked_and_limited():
    results = kb.search("discharge plan follow up", top_k=4)
    scores = [r["score"] for r in results]
    assert len(results) == 4 and scores == sorted(scores, reverse=True)


def test_require_verified_excludes_pending_entries():
    assert kb.search_for_factor("num_medications", require_verified=True) == []


def test_sources_endpoint():
    sources = client.get("/sources").json()
    assert len(sources) == len(data["sources"])
    assert sum(s["excerpts"] for s in sources) == len(excerpts)
    assert all(s["pending_review"] == s["excerpts"] for s in sources)


def test_field_tags_only_reference_real_inputs():
    assert set(FIELD_TAGS) <= set(LABELS)
