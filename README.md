# 30-Day Hospital Readmission Risk Predictor

A live, deployed machine learning API that predicts the risk of a diabetic patient being readmitted to hospital within 30 days — built on the UCI "Diabetes 130-US Hospitals" dataset (101,766 encounters, 1999–2008).

**Live demo:** (https://readmission-risk-api.streamlit.app/)
**API docs:** https://readmission-risk-api-oyvy.onrender.com/docs

---

## What this is

Most work on this dataset stops at a notebook with a reported AUC. This project takes it further — from raw data, through a properly validated model, to a live API and a usable front-end that anyone can test with their own inputs.

## The problem

Hospitals are financially penalized (via CMS's Hospital Readmissions Reduction Program) for excess 30-day readmissions. Predicting which patients are at elevated risk lets hospitals target follow-up care where it matters most.

## Data cleaning decisions

Real hospital data required real decisions, not just default pandas behavior:

- **`max_glu_serum` / `A1Cresult`**: pandas silently treats the literal string `"None"` as missing data on load — but in this dataset, `"None"` means "this test wasn't ordered," a meaningful clinical signal, not an absence of data. Restored as its own category. (These lab fields feed the 110-feature comparison model; the served 14-field model does not use them.)
- **Diagnosis codes (`diag_1/2/3`)**: ~700 raw ICD-9 codes per column were grouped into 9 clinically meaningful categories (Circulatory, Respiratory, Diabetes, etc.) using standard ICD-9 range boundaries, rather than left as high-cardinality raw codes.
- **Death/hospice discharges excluded**: 2,423 encounters ending in death or hospice transfer (`discharge_disposition_id` 11, 13, 14, 19, 20, 21) were removed before encoding and the train/test split, leaving 99,343 encounters, since 30-day readmission isn't clinically possible for these cases — leaving them in would have taught the model a spurious "did this patient die" signal instead of a genuine readmission risk signal.

## Avoiding data leakage

Patients can appear in this dataset multiple times (repeat encounters). A naive random train/test split can place the same patient in both sets, letting the model partially memorize outcomes it will later be "tested" on — inflating reported performance. This project uses `GroupShuffleSplit` on `patient_nbr` to guarantee every patient's records stay entirely in one set or the other.

## Model comparison

Three models were trained and evaluated on the same leak-free split (79,541 training / 19,802 test encounters), using only the 14 fields the API accepts (51 encoded columns):

| Model | AUC |
|---|---|
| Logistic Regression | 0.648 |
| Random Forest | 0.630 |
| **Gradient Boosting (chosen)** | **0.664** |

The served Gradient Boosting model relies mostly on two features: `number_inpatient` (prior inpatient visits, importance 0.552) and `discharge_disposition_id` (0.229). Its highest predicted probability on the test set is 0.782.

*History:* trained on all 110 encoded features (including admission type/source, procedures, outpatient visits, lab results and individual medications), the same models scored Logistic Regression 0.648, Random Forest 0.633 and Gradient Boosting 0.663.

This result closely matches published benchmarks on this exact dataset (independent studies report XGBoost/Gradient Boosting AUC in the 0.63–0.67 range), which suggests ~0.66 reflects a practical ceiling for structured administrative data alone — not a modeling shortfall. Richer inputs (lab trends over time, clinical notes, social determinants of health) are what published work uses to push meaningfully past this range.

## Architecture

```
Streamlit front-end  →  FastAPI /predict, /explain  →  saved Gradient Boosting model
   (user-facing form)      (Render, always-on)           (scikit-learn, .pkl) + SHAP
```

The API accepts 14 input fields, one-hot encodes them, and aligns the result to the model's 51 training columns with `reindex(columns=model_features, fill_value=0)`. Every model input comes from a field the caller sends.

## Input validation

Every request is checked against `model_metadata.json`, which `build_metadata.py` generates from the training split (it reproduces the notebook's split, loads the saved model, and refuses to write if the test AUC no longer matches). Invalid requests get a 422 response whose `detail` names the field and the problem:

| Error `type` | When | Example message |
|---|---|---|
| `invalid_category` | A category value outside the training values, e.g. a typo | `'Femle' is not a valid value for gender. Allowed values: Female, Male` |
| `out_of_range` | A number outside the training min/max | `number_inpatient must be between 0 and 21 (the range seen in training data); got 22` |
| `out_of_scope` | Discharge codes 11, 13, 14, 19, 20, 21 (expired / hospice) | `…excluded from training because 30-day readmission isn't clinically possible, so no prediction is made.` |

Values with fewer than 50 training rows are also rejected, because the model has too little data on them: gender `Unknown/Invalid` (3 rows) and discharge codes 9 (18), 10 (3), 12 (3), 16 (8), 17 (13), 24 (38) and 27 (4). Valid requests get exactly the same response as before. The Streamlit form reads the same file, so it only offers values the API accepts. Discharge disposition labels come from the dataset's own `reference/IDS_mapping.csv`; codes 18 (`NULL`) and 25 (`Not Mapped`) are shown as "Not recorded".

Before this check existed, an unknown category such as `"Femle"` was silently scored as the baseline category with no error.

## Risk tiers

`/predict` returns the original `readmission_risk_score` and `high_risk` fields unchanged, plus `risk_tier`, `risk_percentile` (roughly what share of test-set patients scored lower) and `model_version`. `high_risk` (score above 0.5) is kept for existing callers, but only 72 of 19,802 test encounters cross 0.5, so `risk_tier` is the recommended output.

Tiers are cut at the 50th and 90th percentiles of the model's scores on the test set:

| Tier | Score range | Test encounters | Readmitted within 30 days |
|---|---|---|---|
| Low | below 0.1008 | 9,902 | 6.7% |
| Moderate | 0.1008 to below 0.188 | 7,919 | 13.0% |
| High | 0.188 and above | 1,981 | 26.7% |

The overall test-set readmission rate is 11.2%, so High-tier patients were readmitted at about 2.4 times that rate. Flagging the High tier catches 23.8% of readmissions (precision 26.7%, specificity 91.7%); flagging Moderate or High catches 70.3% (precision 15.7%). The cutoffs and the tier rates come from the same test set, so these rates describe that set rather than an independent validation.

`GET /model-info` returns these numbers along with AUC 0.664 (95% interval 0.650–0.677), a subgroup table, and the allowed input values. Subgroup results by age band, gender and race:

| Group | Test encounters | Readmitted | AUC (95% interval) | Share of readmissions in High tier |
|---|---|---|---|---|
| Age under 50 | 3,193 | 10.9% | 0.706 (0.650–0.750) | 33.5% |
| Age 50–69 | 7,735 | 10.6% | 0.671 (0.647–0.692) | 22.7% |
| Age 70+ | 8,874 | 11.8% | 0.639 (0.623–0.659) | 21.5% |
| Female | 10,645 | 11.2% | 0.671 (0.650–0.693) | 24.9% |
| Male | 9,157 | 11.2% | 0.658 (0.638–0.676) | 22.6% |
| African American | 3,707 | 10.7% | 0.654 (0.625–0.687) | 22.2% |
| Caucasian | 14,886 | 11.5% | 0.663 (0.649–0.681) | 24.1% |

Asian, Hispanic, Other and Unknown race groups each had fewer than 100 readmissions in the test set, so only their counts are reported. The model ranks patients aged 70+ less well than younger patients. All numbers come from `model_metadata.json`, generated by `build_metadata.py`.

## Explanations

`POST /explain` takes the same input as `/predict` and returns the same fields, plus the factors behind the score. It uses SHAP (`TreeExplainer`), which splits the model's output (in log-odds) into one contribution per encoded column. Columns created by one-hot encoding a field are added back together, so there is exactly one factor per input field, and every factor shown is something the user entered. The contributions plus the model's baseline add up to the patient's score; the tests check this.

Each factor has a plain-language description, a direction (raises, lowers, or little effect below 2% of the total), its share of the total contribution, and a `demographic` flag for age, gender and race. A template sentence (no AI involved) summarizes the main factors. For the test set's highest-risk encounter:

> The factors raising this patient's risk the most are 9 inpatient stays in the year before this admission and secondary diagnosis: Neoplasms.

Prior inpatient stays account for 55% of that explanation. The contributions describe how the model arrived at a score, not clinical causes. Computing an explanation takes about 20 ms per request and adds about 30 MB of memory (measured locally, about 210 MB in total).

## Knowledge base

`knowledge_base/sources.json` holds a small, curated set of public sources for the discharge assistant (Stage 5). It has 41 excerpts from 6 sources:

| Source ID | Source |
|---|---|
| `ADA-SOC-2026-S16` | American Diabetes Association, *Standards of Care in Diabetes—2026*, Section 16: Diabetes Care in the Hospital |
| `AHRQ-RED-TOOL1` | AHRQ Re-Engineered Discharge (RED) Toolkit, Tool 1: Overview |
| `STRACK-2014` | Strack et al., *Impact of HbA1c Measurement on Hospital Readmission Rates* (the paper behind this dataset) |
| `CMS-HRRP` | CMS Hospital Readmissions Reduction Program |
| `PSNET-MEDREC` | AHRQ PSNet primer on medication reconciliation |
| `CDC-DSMES-2021` | CDC Diabetes Report Card 2021: Diabetes Self-Management Education and Support (archived page) |

Each excerpt has a short verbatim quote, a plain-language summary, topic tags, and a `verified_by` field. Content is limited to discharge planning, transitions, follow-up, education, medication reconciliation and readmission context; dosing and treatment content is left out because the assistant is advisory only.

**Review status:** the quotes were copied from the source pages, and `python knowledge_base/check_quotes.py` confirms every quote still appears word for word on its page (it needs internet access). The summaries are drafts. Every excerpt is marked `verified_by: "pending"` until a teammate or faculty reviewer checks it. `GET /sources` lists the sources with counts of pending, approved and rejected excerpts.

To review, open `knowledge_base/review_sheet.xlsx` (one row per excerpt, with instructions on the second sheet), fill in Reviewer, Decision (Approve / Edit / Reject), Corrected summary (for Edit) and Notes, then run:

```
python knowledge_base/review_sheet.py apply --dry-run   # preview
python knowledge_base/review_sheet.py apply             # update sources.json
```

Approve sets `verified_by` to the reviewer's name; Edit also replaces the summary and keeps the original draft in the excerpt's `review` record; Reject sets `verified_by` to `"rejected"`, and rejected excerpts are never retrieved. Rows without a decision stay pending. If any row has a problem (no reviewer name, an Edit without a corrected summary, a changed quote, an unknown ID), nothing is written. `python knowledge_base/review_sheet.py export` regenerates the sheet from the current knowledge base. Reading and writing the sheet needs `openpyxl` (in `requirements-dev.txt`).

Retrieval (`knowledge.py`) uses TF-IDF text similarity plus a boost for excerpts tagged with the topics of the patient's risk-raising factors. Gender and race never select guidance. Unrelated questions return no excerpts. Matching is word-based, so a question that uses different words from the source (for example "call the patient" versus "telephone") can miss a relevant excerpt.

## Discharge assistant

`POST /assistant` takes `{"patient": {...14 fields...}, "question": "optional, up to 500 characters"}` and returns discharge-planning items grounded only in the knowledge base. Each request goes through four steps:

1. **Scope check.** Questions asking for a diagnosis, a specific dose, a prescription or a treatment change, or trying to override the rules ("ignore previous instructions"), are refused with an explanation. Questions that merely mention medications ("why do wrong doses happen at discharge?") are allowed.
2. **Retrieval.** With a question, excerpts matching it. Without one, up to three of the patient's main risk-raising factors (never age, gender or race) are each linked to one recommendation and one piece of context that share the factor's topic, plus general discharge items.
3. **Generation.** By default (`LLM_PROVIDER=none`) the assistant is rule-based: each item is the knowledge base's own summary of a cited excerpt. An LLM provider can be plugged in to phrase items, but it only sees the retrieved excerpts and plain-language descriptions of the 14 inputs; email addresses and phone numbers in the question are redacted first.
4. **Checks.** Every item must cite retrieved excerpts. LLM items that cite nothing, cite a source that wasn't provided, or contain dosing/treatment advice are returned under `unsupported` with a reason, not as suggestions. If nothing is supported, or the question isn't covered, the answer is "Insufficient information" rather than a guess. If the LLM fails, times out or returns malformed output, the assistant falls back to rule-based mode and says so in `note`.

Each response includes the patient's risk, the items (`suggestion` or `context`, with the factor they relate to), the cited excerpts with their source, URL, quote and review status, and an advisory-only disclaimer. Citations from excerpts no reviewer has approved yet are labelled **"Source pending review"**.

Settings come from environment variables or a gitignored `.env` file (see `.env.example`): `LLM_PROVIDER` (default `none`), `LLM_API_KEY`, `LLM_MODEL`, `LLM_TIMEOUT_SECONDS` (default 8) and `KB_REQUIRE_VERIFIED` (default `false`; set to `true` once the knowledge base is reviewed, so only approved excerpts are used). No real LLM provider is wired up yet; the LLM path is tested with a fake provider. `GET /assistant/config` shows the active mode and whether a key is set, never the key itself.

## Running the tests

```
pip install -r requirements-dev.txt
pytest
```

The tests cover validation, the response format, known scores, and the Streamlit form (with the API called in-process). The metadata test needs `clean_diabetic_data.csv`, which is not in the repository, so it is skipped elsewhere. After changing the model, run `python build_metadata.py` to regenerate `model_metadata.json`.

## Why the 14-field model

The original model was trained on 110 encoded columns, but the API only receives 14 fields (51 columns). The other 59 columns were filled with 0 at prediction time, which silently assumed things like admission type "0" (not a real code), glucose ">200", A1C ">7" and a dose decrease for insulin and metformin. Three ways of serving predictions were compared on the same test set (2,216 of 19,802 encounters readmitted within 30 days, rate 0.112):

| Option | AUC | Brier | Mean predicted risk | Max score |
|---|---|---|---|---|
| 110-feature model, 59 columns = 0 (previously served) | 0.6626 | 0.0949 | 0.109 | 0.770 |
| 110-feature model, 59 columns = training-set defaults | 0.6618 | 0.0948 | 0.117 | 0.836 |
| **Model retrained on the 14 API fields (now served)** | **0.6642** | **0.0948** | **0.114** | **0.782** |

AUC differences, with 95% intervals from 500 paired bootstrap resamples of test-set patients:

| Comparison | AUC difference | 95% interval |
|---|---|---|
| 14-field minus training defaults | +0.0025 | +0.0010 to +0.0040 |
| Training defaults minus zero-filled | −0.0008 | −0.0024 to +0.0008 |
| 14-field minus zero-filled | +0.0016 | −0.0004 to +0.0035 |

The differences in ranking ability are small; the omitted fields carried little predictive weight. The 14-field model was chosen because every input it uses is something the user actually entered, so explanations of a prediction can never point to a value nobody provided. It also has the best AUC and its average predicted risk (0.114) is closest to the actual rate (0.112). The full comparison is in `exploration.ipynb`.

## A bug worth mentioning

During API testing of an earlier model version, no input could produce a "high risk" flag. Investigation traced this to a real defect: `diag_3_group` had been omitted from the API's input fields entirely — a real predictive signal the API could never receive, no matter what a user entered. Adding it back let high-risk inputs reach scores close to the test set's highest predicted probability. (For the current model, that maximum is 0.782, and the API returns exactly 0.782 for that test encounter.)

## Tech stack

Python, pandas, scikit-learn, SHAP, FastAPI, Uvicorn, Streamlit, deployed on Render (API) and Streamlit Community Cloud (front-end).

## Limitations

- Trained on 1999–2008 administrative data; may not reflect current clinical practice.
- The model uses 14 fields; lab results (A1C, glucose), individual diabetes medications, admission type/source, procedures and outpatient visits are not used.
- AUC ~0.66 reflects real, published performance limits of this dataset — not a target for clinical deployment as-is.
- The assistant's scope and prompt-injection checks are keyword-based. They catch common phrasings (dose requests, "ignore previous instructions") but are not robust against determined attempts, such as rewording, misspellings or other languages. The citation check limits what an LLM's reply can contain, but a clinician still has to review every suggestion.
