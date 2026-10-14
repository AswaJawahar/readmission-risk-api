"""Discharge-planning assistant grounded only in the curated knowledge base.

Pipeline for each request:
1. Scope check: questions asking for a diagnosis, a dose, a prescription or a
   treatment change (or trying to override the rules) are refused.
2. Retrieval: excerpts matching the question, or, with no question, excerpts
   linked to the patient's main risk-raising factors.
3. Generation: rule-based by default (each item is a vetted summary from the
   knowledge base); an LLM provider can phrase items instead, but only from
   the retrieved excerpts.
4. Checks: every item must cite retrieved excerpts. Items that cite nothing,
   cite something that wasn't provided, or drift into dosing/treatment advice
   are listed as unsupported instead of being presented as fact. If nothing
   is supported, the answer is "insufficient information".
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from knowledge import FIELD_TAGS

DISCLAIMER = ("Advisory only: discharge-planning prompts drawn from the cited sources, not a "
              "diagnosis or treatment recommendation. A clinician must review them. Built on "
              "de-identified data as a student proof of concept.")
INSUFFICIENT = ("Insufficient information: the curated sources don't cover this, so no "
                "suggestion is made.")
INSUFFICIENT_UNREVIEWED = ("Insufficient information: no reviewed sources cover this yet "
                           "(only reviewer-approved sources are being used).")
REFUSAL = ("This assistant doesn't advise on diagnoses, medication doses, prescriptions or "
           "treatment changes; those decisions belong to the treating clinician. It can help "
           "with discharge planning, such as follow-up, patient education and medication "
           "reconciliation.")

# Requests for (or statements of) specific doses, prescriptions, treatment changes or
# diagnoses. Mentions such as "wrong doses happen at discharge" are allowed.
OUT_OF_SCOPE_PATTERNS = [
    r"\b(what|which|right|correct|starting|new|higher|lower) (dose|dosage)\b",
    r"\b(dose|dosage|dosing) (of|for)\b",
    r"\bhow (much|many units)\b.{0,30}\b(insulin|metformin|medications?|medicines?|drugs?)\b",
    r"\b\d+\s*(units?|mg)\b",
    r"\bprescri(be|bed|bing)\b",
    r"\btitrat",
    r"\b(increase|decrease|raise|lower|reduce|stop|start|switch|discontinue)\b.{0,40}"
    r"\b(insulin|metformin|medications?|medicines?|drugs?)\b",
    r"\bdiagnos(e|ing)\b",
    r"\bdoes (he|she|the patient|this patient) have\b",
]
INJECTION_PATTERNS = [
    r"\bignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules)",
    r"\bsystem prompt\b",
    r"\bjailbreak\b",
]
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")

MAX_SUGGESTIONS = 5
MAX_CONTEXT = 2
FACTORS_TO_LINK = 3
GENERAL_QUERY = "structured discharge plan and follow-up appointments scheduled before discharge"

SYSTEM_PROMPT = """You help hospital discharge staff plan the discharge of a patient with diabetes.
Rules:
- Use ONLY the numbered excerpts provided. Do not add facts, numbers or advice from anywhere else.
- Every item must cite one or more excerpt IDs from the list in "citations".
- Do not give diagnoses, medication doses, prescriptions or treatment changes.
- If the excerpts do not answer the question, return no items and set "insufficient_information" to true.
- Write in plain language for nurses and case managers.
Reply with JSON only, in this shape:
{"items": [{"text": "...", "type": "suggestion" or "context", "citations": ["EXCERPT-ID"]}],
 "insufficient_information": false}"""


def matches_any(patterns, text):
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def redact(text):
    """Removes email addresses and phone numbers before anything leaves the server."""
    return PHONE.sub("[redacted phone]", EMAIL.sub("[redacted email]", text))


def review_label(verified_by):
    return "Source pending review" if verified_by == "pending" else f"Reviewed by {verified_by}"


class ProviderError(Exception):
    pass


# Real providers are registered here once one is chosen; each takes Settings and
# has complete(system, user) -> str. Until then only rule-based mode is available.
PROVIDERS = {}


def make_provider(settings):
    """Returns (provider or None, note explaining the mode)."""
    name = settings.llm_provider
    if name == "none":
        return None, "Rule-based mode (no LLM provider configured)."
    if name not in PROVIDERS:
        return None, f"LLM provider '{name}' is not available; using rule-based mode."
    if not settings.llm_api_key:
        return None, f"LLM_API_KEY is not set for provider '{name}'; using rule-based mode."
    return PROVIDERS[name](settings), f"LLM provider '{name}'."


class Assistant:
    def __init__(self, knowledge, settings, provider=None, note=""):
        self.kb = knowledge
        self.settings = settings
        self.provider = provider
        self.note = note
        self._executor = ThreadPoolExecutor(max_workers=4)

    # ---- retrieval -------------------------------------------------------
    def _retrieve(self, factors, question):
        require = self.settings.kb_require_verified
        if question:
            return self.kb.search(question, top_k=MAX_SUGGESTIONS + MAX_CONTEXT,
                                  require_verified=require), {}

        candidates, linked = [], {}

        def add(results, factor=None):
            for r in results:
                if r["excerpt_id"] not in linked:
                    linked[r["excerpt_id"]] = factor
                    candidates.append(r)

        raising = [f for f in factors if f["direction"] == "raises" and not f["demographic"]]
        for factor in raising[:FACTORS_TO_LINK]:
            # Only excerpts that share a topic with the factor count as linked to it
            topics = set(FIELD_TAGS.get(factor["field"], []))
            results = [r for r in self.kb.search_for_factor(factor["field"], top_k=len(self.kb.excerpts),
                                                            require_verified=require)
                       if r["excerpt_id"] not in linked and topics & set(r["tags"])]
            # Per factor: the best actionable recommendation, then one piece of context
            add([r for r in results if r["kind"] == "recommendation"][:1], factor)
            add([r for r in results if r["kind"] != "recommendation"][:1], factor)
        add(self.kb.search(GENERAL_QUERY, tags=["care_transitions", "follow_up"], top_k=2,
                           require_verified=require))
        return candidates, linked

    # ---- generation ------------------------------------------------------
    def _rule_based(self, candidates, linked):
        suggestions = [c for c in candidates if c["kind"] == "recommendation"][:MAX_SUGGESTIONS]
        context = [c for c in candidates if c["kind"] != "recommendation"][:MAX_CONTEXT]
        return [self._item(c["summary"], "suggestion" if c in suggestions else "context",
                           [c["excerpt_id"]], linked.get(c["excerpt_id"]))
                for c in suggestions + context]

    @staticmethod
    def _item(text, kind, citations, factor=None):
        return {"text": text, "type": kind, "citations": citations,
                "linked_factor": factor["description"] if factor else None}

    def _prompt(self, explanation, question, candidates):
        patient = {
            "risk_tier": explanation["risk_tier"],
            "readmission_risk_score": explanation["readmission_risk_score"],
            # Plain-language descriptions of the 14 model inputs only; no identifiers exist
            "factors": [{"description": f["description"], "effect": f["direction"]}
                        for f in explanation["factors"]],
        }
        excerpts = [{"id": c["excerpt_id"], "source": c["source"]["title"],
                     "summary": c["summary"], "quote": c["quote"]} for c in candidates]
        request = {"patient": patient, "excerpts": excerpts,
                   "question": redact(question) if question else
                   "What discharge-planning items should staff address for this patient?"}
        return SYSTEM_PROMPT, json.dumps(request, ensure_ascii=False)

    def _call_provider(self, system, user):
        future = self._executor.submit(self.provider.complete, system, user)
        try:
            return future.result(timeout=self.settings.llm_timeout_seconds)
        except FutureTimeout:
            raise ProviderError(f"no reply within {self.settings.llm_timeout_seconds:g} s")
        except Exception as exc:  # any provider failure falls back to rule-based mode
            raise ProviderError(f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _parse(reply):
        try:
            data = json.loads(reply)
        except (TypeError, json.JSONDecodeError):
            raise ProviderError("reply was not valid JSON")
        if not isinstance(data, dict) or not isinstance(data.get("items", []), list):
            raise ProviderError("reply did not match the expected format")
        return data

    def _check_llm_items(self, data, candidates):
        allowed = {c["excerpt_id"] for c in candidates}
        items, unsupported = [], []
        for raw in data.get("items", []):
            raw = raw if isinstance(raw, dict) else {}
            text = str(raw.get("text", "")).strip()
            citations = raw.get("citations") if isinstance(raw.get("citations"), list) else []
            citations = [str(c) for c in citations]
            if not text:
                continue
            if not citations:
                unsupported.append({"text": text, "reason": "No source was cited."})
            elif not set(citations) <= allowed:
                unsupported.append({"text": text, "reason": "Cites a source that was not provided: "
                                    + ", ".join(sorted(set(citations) - allowed))})
            elif matches_any(OUT_OF_SCOPE_PATTERNS, text):
                unsupported.append({"text": text, "reason": "Outside the assistant's advisory scope "
                                    "(dosing, prescribing, diagnosis or treatment change)."})
            else:
                kind = raw.get("type") if raw.get("type") in {"suggestion", "context"} else "suggestion"
                items.append(self._item(text, kind, citations))
        return items[:MAX_SUGGESTIONS + MAX_CONTEXT], unsupported

    # ---- response --------------------------------------------------------
    def respond(self, explanation, question=None):
        question = (question or "").strip()
        base = {
            "risk": {k: explanation[k] for k in ("readmission_risk_score", "risk_tier", "risk_percentile")},
            "question": question or None,
            "disclaimer": DISCLAIMER,
            "kb_require_verified": self.settings.kb_require_verified,
            "note": self.note, "items": [], "unsupported": [], "citations": {},
        }
        if question and matches_any(INJECTION_PATTERNS + OUT_OF_SCOPE_PATTERNS, question):
            return {**base, "mode": "none", "status": "refused", "message": REFUSAL}

        candidates, linked = self._retrieve(explanation["factors"], question)
        if not candidates:
            message = INSUFFICIENT_UNREVIEWED if self.settings.kb_require_verified else INSUFFICIENT
            return {**base, "mode": "none", "status": "insufficient_information", "message": message}

        mode, note, unsupported = "rule_based", self.note, []
        items = None
        if self.provider is not None:
            try:
                data = self._parse(self._call_provider(*self._prompt(explanation, question, candidates)))
                items, unsupported = self._check_llm_items(data, candidates)
                mode = "llm"
            except ProviderError as exc:
                mode, note = "rule_based_fallback", f"LLM unavailable ({exc}); used rule-based mode."
        if items is None:
            items = self._rule_based(candidates, linked)

        by_id = {c["excerpt_id"]: c for c in candidates}
        cited = sorted({cid for item in items for cid in item["citations"]})
        citations = {cid: {"title": by_id[cid]["source"]["title"], "url": by_id[cid]["source"]["url"],
                           "section": by_id[cid]["section"], "quote": by_id[cid]["quote"],
                           "verified_by": by_id[cid]["verified_by"],
                           "review_label": review_label(by_id[cid]["verified_by"])}
                     for cid in cited}

        if not items:
            return {**base, "mode": mode, "note": note, "status": "insufficient_information",
                    "message": INSUFFICIENT, "unsupported": unsupported}
        message = ("From the curated sources, about your question:" if question else
                   "Discharge-planning items from the curated sources, linked to this patient's main risk factors:")
        return {**base, "mode": mode, "note": note, "status": "answered", "message": message,
                "items": items, "unsupported": unsupported, "citations": citations}
