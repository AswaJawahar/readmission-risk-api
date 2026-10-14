# CLAUDE.md

AI-assisted discharge support for diabetic patients (course project, Lab 4): a FastAPI service
(`main.py`, deployed on Render) and a Streamlit app (`app.py`, deployed on Streamlit Community
Cloud), both deployed from `master`. Work happens on `ai-discharge-support`.

## Standing rules

- **Ask before** any git merge, push, dependency install, retraining, or deleting files/data.
  Approval for one action doesn't carry over to the next.
- **Never send the user's email or any personal details to external services** (API calls,
  lookups, query parameters), even if a service asks for one.
- **Never invent metrics.** Every number shown or written must be computed from the real model and
  test set (`model_metadata.json`, built by `build_metadata.py`). Any factor shown to users must be a
  real model input.
- **API keys only via `LLM_API_KEY` in a gitignored `.env`** (never `ANTHROPIC_API_KEY`, never committed).
  The app must keep working without a key (rule-based mode).
- **One stage at a time:** build, run the tests, show a diff summary and test output, commit on
  `ai-discharge-support`, then wait for the go-ahead.
- Keep `/predict` backward compatible (same fields; new response fields are additions only).
- No "HIPAA-compliant" or similar claims. Label data as de-identified/synthetic and the tool as advisory only.

## Stages

| Stage | Status |
|---|---|
| Task 1: retrain with the death/hospice filter before encoding, docs, pins | Done, on master |
| 14-field model (replaces zero-filled 110-feature serving) | Done, on master |
| 1. Input validation (categories, ranges, out-of-scope discharge codes) | Done, on master |
| 2. Risk tiers and test-set performance metrics | Done, on master |
| 3. SHAP explanations with plain-language labels | Done, on master |
| 4. Curated knowledge base, retrieval, review spreadsheet | Done, on master (all excerpts still `pending` review) |
| 5. `/assistant`, grounded and cited, with guardrails | Done, on master (rule-based; no real LLM provider yet) |
| 6. Streamlit redesign: patient queue, dashboard, What-if, About | Done, on master |
| 7. Metadata-only audit logging and mock role-based access | Next |

## Key decisions

- **14-field model.** Trained only on the 14 fields the API accepts, so explanations never point at a
  value nobody entered. Test AUC 0.6642 (`EXPECTED_TEST_AUC` in `build_metadata.py`).
- **Risk tiers at the 50th / 90th percentiles of test-set scores** (cutoffs 0.1008 / 0.188). The
  Moderate tier starts below the 11.2% average rate; the UI explains this.
- **Categories with fewer than 50 training rows are rejected** by the API (listed in `model_metadata.json`).
- **`KB_REQUIRE_VERIFIED` stays off until the knowledge base is reviewed**; citations from unreviewed
  excerpts are labelled "Source pending review".
- **Rule-based assistant by default** (`LLM_PROVIDER=none`). The LLM path is built and tested with a fake
  provider only; the user will choose a real provider later.
- Factors are worded as "associated with a higher/lower score", never "raises/lowers risk".
- The UI leads with the tier and a multiple of the average rate; the raw score is secondary.

## Practical notes

- `clean_diabetic_data.csv` is gitignored. Scripts that need it run locally only:
  `python build_metadata.py` (regenerate `model_metadata.json` after any model change),
  `python build_patient_queue.py` (regenerate `demo/patient_queue.json`). Tests that need it are skipped elsewhere.
- Tests: `venv/Scripts/python.exe -m pytest` (dev deps in `requirements-dev.txt`).
- Running `exploration.ipynb` end to end retrains and overwrites the `.pkl` files — that counts as retraining.
- Knowledge-base review: `knowledge_base/review_sheet.xlsx`, then `python knowledge_base/review_sheet.py apply --dry-run`
  / `apply`. Don't re-export the sheet while someone may have unsaved edits (check `git diff` and Excel's
  `~$` lock file first). `knowledge_base/check_quotes.py` re-checks quotes against the live pages.
- Render uses `.python-version` (3.11). Versions are pinned in `requirements.txt`, including numba/llvmlite for shap.
- The Streamlit app reads `API_URL` (environment or secrets), defaulting to the Render URL.
- Streamlit tables draw on a canvas: AppTest can't click rows, so tests open patients via
  `session_state["selected_patient"]`; real clicks were checked in headless Chrome.
