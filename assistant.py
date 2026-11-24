"""Discharge-planning assistant grounded only in the curated knowledge base.

Pipeline for each request:
1. Scope check: questions asking for a diagnosis, a dose, a prescription or a
   treatment change (or trying to override the rules) are refused.
2. Retrieval: excerpts matching the question, or, with no question, excerpts
   linked to the factors most associated with a higher score for this patient.
3. Generation: rule-based by default (each item is a vetted summary from the
   knowledge base); an LLM provider (Gemini) can phrase items instead, but only
   from the retrieved excerpts. The LLM never receives patient data: its request
   holds the excerpts and the redacted question, nothing else. Excerpts are
   matched to the patient's factors here, on the server, before and after the call.
4. Checks: every item must cite retrieved excerpts. Items that cite nothing,
   cite something that wasn't provided, or drift into dosing/treatment advice
   are listed as unsupported instead of being presented as fact. If nothing
   is supported, the answer is "insufficient information".
"""
import json
import re
import time
import urllib.error
import urllib.request
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
PATIENT_ID = re.compile(r"\bPT-?\d+\b", flags=re.IGNORECASE)

MAX_SUGGESTIONS = 5
MAX_CONTEXT = 2
FACTORS_TO_LINK = 3
GENERAL_QUERY = "structured discharge plan and follow-up appointments scheduled before discharge"

NO_QUESTION = "Which discharge-planning items do these excerpts support?"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_DEFAULT_MODEL = "gemini-3.6-flash"  # override with LLM_MODEL
# HTTP 503 means Google's servers are briefly overloaded, so it gets one retry after a short
# pause, if enough of the time limit is left. Quota errors (429) and others are not retried.
RETRY_STATUS = 503
RETRY_DELAY_SECONDS = 1.0
MIN_RETRY_SECONDS = 1.0  # time that must remain after the pause for a retry to be worth it

SYSTEM_PROMPT = """You help hospital discharge staff plan discharges for patients with diabetes.
You are not given any information about an individual patient, and you must not ask for it or guess it.
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
    """Removes email addresses, phone numbers and patient IDs before anything leaves the server."""
    text = PATIENT_ID.sub("[redacted patient id]", text)
    return PHONE.sub("[redacted phone]", EMAIL.sub("[redacted email]", text))


def review_label(verified_by):
    return "Source pending review" if verified_by == "pending" else f"Reviewed by {verified_by}"


class ProviderError(Exception):
    pass


def normalize_model_name(name):
    """'Gemini 3.5 Flash Lite' or 'models/gemini-3.5-flash-lite' -> 'gemini-3.5-flash-lite'."""
    return re.sub(r"[\s_]+", "-", name.strip().lower().removeprefix("models/"))


class GeminiProvider:
    """Google Gemini over its REST API (standard library only, no extra dependency).

    The key travels in a request header, never in the URL, and error messages
    carry only the HTTP status and Google's error code.
    """

    def __init__(self, settings):
        self.api_key = settings.llm_api_key
        self.model = normalize_model_name(settings.llm_model or GEMINI_DEFAULT_MODEL)
        # The model name goes into the URL path, so only plain model ids are accepted
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", self.model):
            raise ValueError(f"LLM_MODEL '{settings.llm_model}' is not a valid model name")
        self.timeout = settings.llm_timeout_seconds

    def complete(self, system, user):
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }
        request = urllib.request.Request(
            GEMINI_URL.format(model=self.model), data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key})
        deadline = time.monotonic() + self.timeout
        timeout, retried = self.timeout, False
        while True:
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                remaining = deadline - time.monotonic() - RETRY_DELAY_SECONDS
                if exc.code == RETRY_STATUS and not retried and remaining >= MIN_RETRY_SECONDS:
                    time.sleep(RETRY_DELAY_SECONDS)
                    timeout, retried = remaining, True
                    continue
                after = " after one retry" if retried else ""
                raise ProviderError(f"Gemini returned HTTP {exc.code}{self._error_code(exc)}{after}") from None
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                raise ProviderError(f"Gemini could not be reached ({type(exc).__name__})") from None
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        except (KeyError, IndexError, TypeError, AttributeError):
            raise ProviderError("Gemini returned no text") from None
        if not text.strip():
            raise ProviderError("Gemini returned no text")
        # Strip a ```json fence if the model added one despite the JSON response type
        return re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text)

    @staticmethod
    def _error_code(exc):
        try:
            return f" ({json.loads(exc.read().decode('utf-8'))['error']['status']})"
        except Exception:
            return ""


# Each provider takes Settings and has complete(system, user) -> str.
PROVIDERS = {"gemini": GeminiProvider}


def make_provider(settings):
    """Returns (provider or None, note explaining the mode)."""
    name = settings.llm_provider
    if name == "none":
        return None, "Rule-based mode (no LLM provider configured)."
    if name not in PROVIDERS:
        return None, f"LLM provider '{name}' is not available; using rule-based mode."
    if not settings.llm_api_key:
        return None, f"LLM_API_KEY is not set for provider '{name}'; using rule-based mode."
    try:
        provider = PROVIDERS[name](settings)
    except ValueError as exc:
        return None, f"{exc}; using rule-based mode."
    model = getattr(provider, "model", "")
    return provider, f"LLM provider '{name}'" + (f" (model {model})." if model else ".")


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

        higher = [f for f in factors if f["direction"] == "higher" and not f["demographic"]]
        for factor in higher[:FACTORS_TO_LINK]:
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

    def _prompt(self, question, candidates):
        """The whole request sent to the LLM: excerpts and the redacted question only.

        Deliberately takes no patient data (no fields, factors, score or tier), so
        none can reach the provider.
        """
        excerpts = [{"id": c["excerpt_id"], "source": c["source"]["title"],
                     "summary": c["summary"], "quote": c["quote"]} for c in candidates]
        request = {"excerpts": excerpts, "question": redact(question) if question else NO_QUESTION}
        return SYSTEM_PROMPT, json.dumps(request, ensure_ascii=False)

    def _call_provider(self, system, user):
        future = self._executor.submit(self.provider.complete, system, user)
        try:
            return future.result(timeout=self.settings.llm_timeout_seconds)
        except FutureTimeout:
            raise ProviderError(f"no reply within {self.settings.llm_timeout_seconds:g} s")
        except ProviderError:
            raise
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

    def _check_llm_items(self, data, candidates, linked):
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
                # The factor link is made here from the cited excerpts; the LLM never saw the factors
                factor = next((linked[c] for c in citations if linked.get(c)), None)
                items.append(self._item(text, kind, citations, factor))
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
                data = self._parse(self._call_provider(*self._prompt(question, candidates)))
                items, unsupported = self._check_llm_items(data, candidates, linked)
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
                   "Discharge-planning items from the curated sources, linked to the factors behind this patient's score:")
        return {**base, "mode": mode, "note": note, "status": "answered", "message": message,
                "items": items, "unsupported": unsupported, "citations": citations}
