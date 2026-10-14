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
Streamlit front-end  →  FastAPI /predict endpoint  →  saved Gradient Boosting model
   (user-facing form)      (Render, always-on)          (scikit-learn, .pkl)
```

The API accepts 14 input fields (Pydantic checks field names and types), one-hot encodes them, and aligns the result to the model's 51 training columns with `reindex(columns=model_features, fill_value=0)`. Every model input comes from a field the caller sends.

**Unknown categories are silently treated as the baseline.** A category value the model never saw (including a typo such as `"Femle"`) produces a column that `reindex` drops, so the row looks identical to the reference category dropped during training (`drop_first=True`): race → AfricanAmerican, gender → Female, age → [0-10), diagnosis group → Circulatory, diabetesMed → No. No error is raised.

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

Python, pandas, scikit-learn, FastAPI, Uvicorn, Streamlit, deployed on Render (API) and Streamlit Community Cloud (front-end).

## Limitations

- Trained on 1999–2008 administrative data; may not reflect current clinical practice.
- The model uses 14 fields; lab results (A1C, glucose), individual diabetes medications, admission type/source, procedures and outpatient visits are not used.
- The API does not yet reject death/hospice discharge codes or unknown category values; only the Streamlit front-end blocks death/hospice codes.
- AUC ~0.66 reflects real, published performance limits of this dataset — not a target for clinical deployment as-is.
