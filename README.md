# Hospital Readmission Prediction from Clinical Records

Predicting 30-day hospital readmission for diabetic patients using the UCI *Diabetes 130-US Hospitals* dataset (101,766 encounters / 71,518 patients / 50 features). A leakage-safe pipeline (patient-level aggregation → 95/5 split → EDA-driven feature engineering) is compared across one-vs-rest logistic regression, Gaussian naive Bayes, LDA, decision tree, random forest, CatBoost, LightGBM and XGBoost.

## Why this dataset is hard

- **Patient-level leakage.** The same patient appears in multiple encounters. Treating encounters as independent rows inflates performance — an identical patient can land in both train and test. Fixed here by aggregating encounters per patient *before* splitting.
- **Severe class imbalance.** `NO` ≈ 53%, `>30` ≈ 34%, `<30` ≈ 11%. Under weighted metrics this makes the minority class nearly invisible; a model that never predicts `<30` still scores ~0.78 accuracy.
- **`?` instead of `NaN`.** The raw CSV encodes missing values as the literal string `?`, so `df.isnull()` reports zero missing values until they are converted.
- **High-cardinality categoricals.** `diag_1/2/3` have 848 / 923 / 954 unique ICD-9 codes, and `discharge_disposition_id` has 30 levels — both unusable without domain-driven grouping.

## Pipeline

| Step | What was done | Rationale |
|---|---|---|
| Missing values | `?` → `NaN`; dropped `weight` (96.9% NA), `medical_specialty` (49.1%), `payer_code` (39.6%); mode-imputed the rest | Three columns carry no usable signal; mode imputation is defensible for low-missingness categoricals |
| Patient aggregation | Summed 8 utilisation features per `patient_nbr`; added `number_visits`; kept latest categorical values | Removes patient-level leakage and encodes cumulative burden |
| Split | 95% / 5% held-out, `random_state=42`, **before** any EDA | Prevents EDA-driven decisions from leaking into the test set |
| Admission type | 8 → 3 (`Emergency` / `Elective` / `Unknown`) | Emergency, Urgent and Trauma Center behave identically |
| Discharge disposition | 30 → 7 | 30 levels are unusable; grouping preserves clinically meaningful outcomes |
| Admission source | 25 → 5 | Most levels are rare administrative artefacts |
| Diagnoses | ICD-9 codes → 9 disease groups (circulatory, digestive, genitourinary, diabetes, injury, musculoskeletal, neoplasms, respiratory, other) | Turns 848+ raw codes into 9 interpretable features |
| Medications | 23 → 4 (`metformin`, `glipizide`, `glyburide`, `insulin`, kept at >10% prescription rate); built `number_changes` | 19 drugs are prescribed to almost no one; dose changes captured in one feature |

The final design matrix has 25 variables (17 categorical → one-hot, 8 numerical).

## Results

95% train / 5% test, weighted precision/recall/F1 (weights = class frequency).

| Model | Test Acc | Test Prec | Test Rec | Test F1 |
|---|---|---|---|---|
| Logistic regression (one-vs-rest) | 0.78 | 0.78 | 0.78 | 0.74 |
| CatBoost (`iterations=1000`, `lr=0.1`) | 0.78 | 0.77 | 0.78 | **0.75** |
| LightGBM | 0.78 | 0.76 | 0.78 | 0.75 |
| XGBoost (`max_depth=5`) | 0.78 | 0.76 | 0.78 | 0.74 |
| Random forest | 0.78 | 0.75 | 0.78 | 0.74 |
| Gaussian naive Bayes | 0.76 | 0.74 | 0.76 | 0.73 |
| Decision tree | 0.72 | 0.70 | 0.72 | 0.71 |
| LDA | 0.74 | 0.73 | 0.74 | 0.69 |

**The headline number is the trap.** Overall accuracy 0.78 looks respectable, but per-class recall tells the real story:

| Class | Logistic regression recall | CatBoost recall |
|---|---|---|
| `NO` (no readmission) | **0.99** | **0.98** |
| `>30` days | 0.58 | 0.59 |
| `<30` days | **0.01** | **0.06** |

Both models are effectively a majority-class detector. The `<30` class contains roughly 200 test cases and the models recover 9 (LR) / 27 (CatBoost) of them. Any clinical deployment targetting early readmission would need class weighting, threshold tuning, or a two-stage design — accuracy is the wrong objective for this task.

An interesting secondary finding: the gradient-boosting models recover a *higher* share of the minority `<30` class than logistic regression while matching its accuracy, suggesting that tree ensembles handle the imbalanced, interacting structure better even when aggregate metrics cannot distinguish them.

## Repository contents

```
hospital-readmission-prediction/
├── model.ipynb           # Full analysis: preprocessing → EDA → modelling → evaluation (132 cells)
├── requirements.txt      # Python dependencies
├── figures/              # Exported plots referenced in the report
└── 614_report.pdf        # Written report: motivation, EDA narrative, discussion
```

## Reproducing

```bash
pip install -r requirements.txt
jupyter notebook model.ipynb
```

Then place `diabetic_data.csv` in the working directory. The notebook reads it from a relative path (`diabetic_data.csv`), so run it from the project root.

**Data.** The dataset is **not** included in this repository. Download `dataset_diabetes.zip` from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), unzip, and place `diabetic_data.csv` next to the notebook.

## Reference

Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International, 2014, Article ID 781670.

## Authors

Xingyu Zhu, Yang Ye, Zhenggang Tan — EN.553.614, Johns Hopkins University.

## License

Code released under the MIT License. The dataset retains its original UCI terms of use.

