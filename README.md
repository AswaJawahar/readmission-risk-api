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
- **Death/hospice discharges excluded**: ~2,400 encounters ending in death or hospice transfer were removed before training, since 30-day readmission isn't clinically possible for these cases — leaving them in would have taught the model a spurious "did this patient die" signal instead of a genuine readmission risk signal.

## Avoiding data leakage

Patients can appear in this dataset multiple times (repeat encounters). A naive random train/test split can place the same patient in both sets, letting the model partially memorize outcomes it will later be "tested" on — inflating reported performance. This project uses `GroupShuffleSplit` on `patient_nbr` to guarantee every patient's records stay entirely in one set or the other.

## Model comparison

Three models were trained and evaluated on the same leak-free test set:

| Model | AUC |
|---|---|
| Logistic Regression | 0.648 |
| Random Forest | 0.632 |
| **Gradient Boosting (chosen)** | **0.663** |

This result closely matches published benchmarks on this exact dataset (independent studies report XGBoost/Gradient Boosting AUC in the 0.63–0.67 range), which suggests ~0.66 reflects a practical ceiling for structured administrative data alone — not a modeling shortfall. Richer inputs (lab trends over time, clinical notes, social determinants of health) are what published work uses to push meaningfully past this range.

## Architecture

```
Streamlit front-end  →  FastAPI /predict endpoint  →  saved Gradient Boosting model
   (user-facing form)      (Render, always-on)          (scikit-learn, .pkl)
```

The API defines a strict input contract (Pydantic) and reconstructs every request into the model's exact trained 110-column feature shape via `reindex(..., fill_value=0)` — regardless of what the caller sends. This means the underlying model can be retrained or extended with new features without changing the serving logic.

## A bug worth mentioning

During API testing, no input could produce a "high risk" flag, despite the model being capable of AUC 0.663. Investigation traced this to a real defect: `diag_3_group` had been omitted from the API's input fields entirely — a real predictive signal the API could never receive, no matter what a user entered. Adding it back moved a test case's risk score from 0.585 to 0.704, reproducing the test set's actual highest score almost exactly.

## Tech stack

Python, pandas, scikit-learn, FastAPI, Uvicorn, Streamlit, deployed on Render (API) and Streamlit Community Cloud (front-end).

## Limitations

- Trained on 1999–2008 administrative data; may not reflect current clinical practice.
- Uses a simplified 13-field input set (the features with the most model influence) rather than the full 110-column training feature set.
- AUC ~0.66 reflects real, published performance limits of this dataset — not a target for clinical deployment as-is.
