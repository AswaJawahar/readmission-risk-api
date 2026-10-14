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

- **`max_glu_serum` / `A1Cresult`**: pandas silently treats the literal string `"None"` as missing data on load — but in this dataset, `"None"` means "this test wasn't ordered," a meaningful clinical signal, not an absence of data. Restored as its own category.
- **Diagnosis codes (`diag_1/2/3`)**: ~700 raw ICD-9 codes per column were grouped into 9 clinically meaningful categories (Circulatory, Respiratory, Diabetes, etc.) using standard ICD-9 range boundaries, rather than left as high-cardinality raw codes.
- **Death/hospice discharges excluded**: 2,423 encounters ending in death or hospice transfer (`discharge_disposition_id` 11, 13, 14, 19, 20, 21) were removed before encoding and the train/test split, leaving 99,343 encounters, since 30-day readmission isn't clinically possible for these cases — leaving them in would have taught the model a spurious "did this patient die" signal instead of a genuine readmission risk signal.

## Avoiding data leakage

Patients can appear in this dataset multiple times (repeat encounters). A naive random train/test split can place the same patient in both sets, letting the model partially memorize outcomes it will later be "tested" on — inflating reported performance. This project uses `GroupShuffleSplit` on `patient_nbr` to guarantee every patient's records stay entirely in one set or the other.

## Model comparison

Three models were trained and evaluated on the same leak-free split (79,541 training / 19,802 test encounters, 110 encoded features):

| Model | AUC |
|---|---|
| Logistic Regression | 0.648 |
| Random Forest | 0.633 |
| **Gradient Boosting (chosen)** | **0.663** |

The Gradient Boosting model relies mostly on two features: `number_inpatient` (prior inpatient visits, importance 0.544) and `discharge_disposition_id` (0.216). Its highest predicted probability on the test set is 0.776.

This result closely matches published benchmarks on this exact dataset (independent studies report XGBoost/Gradient Boosting AUC in the 0.63–0.67 range), which suggests ~0.66 reflects a practical ceiling for structured administrative data alone — not a modeling shortfall. Richer inputs (lab trends over time, clinical notes, social determinants of health) are what published work uses to push meaningfully past this range.

## Architecture

```
Streamlit front-end  →  FastAPI /predict endpoint  →  saved Gradient Boosting model
   (user-facing form)      (Render, always-on)          (scikit-learn, .pkl)
```

The API accepts 14 input fields (Pydantic checks field names and types), one-hot encodes them, and aligns the result to the model's 110 training columns with `reindex(columns=model_features, fill_value=0)`. In practice this means:

- **Unknown categories are silently treated as the baseline.** A category value the model never saw (including a typo such as `"Femle"`) produces a column that `reindex` drops, so the row looks identical to the reference category dropped during training (`drop_first=True`): race → AfricanAmerican, gender → Female, age → [0-10), diagnosis group → Circulatory, diabetesMed → No. No error is raised.
- **The 59 model columns that come from fields the API doesn't accept are filled with 0.** The 14 inputs cover 51 of the 110 columns. For the omitted numeric fields `admission_type_id` and `admission_source_id`, 0 lies outside the training range (minimum 1); `num_procedures` and `number_outpatient` become 0. For omitted categorical fields, all-zero dummies encode the reference category: `max_glu_serum` ">200", `A1Cresult` ">7", `change` "Ch", dose "Down" for the 13 medications that have that level (including insulin and metformin), and "No" for the rest. Every API prediction is therefore made as if those values applied.

## A bug worth mentioning

During API testing of an earlier model version, no input could produce a "high risk" flag. Investigation traced this to a real defect: `diag_3_group` had been omitted from the API's input fields entirely — a real predictive signal the API could never receive, no matter what a user entered. Adding it back let high-risk inputs reach scores close to the test set's highest predicted probability. (For the current model, that maximum is 0.776.)

## Tech stack

Python, pandas, scikit-learn, FastAPI, Uvicorn, Streamlit, deployed on Render (API) and Streamlit Community Cloud (front-end).

## Limitations

- Trained on 1999–2008 administrative data; may not reflect current clinical practice.
- The API takes a simplified 14-field input set rather than the full 110-column training feature set; the missing columns are zero-filled as described under Architecture, which affects every prediction.
- The API does not yet reject death/hospice discharge codes or unknown category values; only the Streamlit front-end blocks death/hospice codes.
- AUC ~0.66 reflects real, published performance limits of this dataset — not a target for clinical deployment as-is.
