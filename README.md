# MediPulse.ai – Data

Synthetic MIMIC-like ICU data generation and preprocessing.

## 1. Generate synthetic data

From the **project root** (`MediPulse.ai`):

```bash
pip install -r requirements.txt
python -m data.generate_synthetic_mimic --n-patients 500 --seed 42
```

- **Output:** `data/raw/`  
  - `patients.csv`, `admissions.csv`, `icustays.csv`  
  - `chartevents.csv` (vitals), `labevents.csv`, `diagnoses_icd.csv`  
- **Options:** `--n-patients`, `--seed`, `--out-dir`

## 2. Preprocess

```bash
python -m data.preprocess --raw-dir data/raw --out-dir data/processed
```

- **Output:** `data/processed/`  
  - `chartevents_cleaned.csv`, `labevents_cleaned.csv`  
  - `stay_vitals.csv`, `stay_labs.csv`, `stay_labels.csv`  
  - `stays_merged.csv` – one row per ICU stay with vitals, labs, and labels  
  - `stays_featured.csv` – same stays with **engineered features** (see below)

## 3. Feature engineering

Features are computed automatically when you run `preprocess`, or run standalone:

```bash
python -m data.feature_engineering --processed-dir data/processed --raw-dir data/raw
```

**Engineered features:**

| Feature group   | Columns |
|-----------------|--------|
| **Age**         | `age_at_admission`, `anchor_age` |
| **Diagnosis**   | `n_diagnoses`, and one-hot flags: `dx_hypertension`, `dx_diabetes`, `dx_pneumonia`, `dx_heart_failure`, `dx_aki`, `dx_gerd`, `dx_copd`, `dx_cad`, `dx_dyspnea`, `dx_sepsis`, `dx_severe_sepsis` |
| **Prior visits**| `prior_admissions`, `prior_icu_visits` (count of previous hospital/ICU stays for same patient) |
| **Lab values**  | From `stays_merged`: e.g. `glucose_mean`, `glucose_first`, `creatinine_mean`, `lactate_mean`, `hemoglobin_mean`, etc. |
| **Length of stay** | `los_icu_hours`, `los_hospital_hours` |

Output: `data/processed/stays_featured.csv` (use for modeling).

## Schema

- **Clinical schema:** `data/clinical_schema.py` (chart/lab itemids, ranges, admission types, ICD samples).  
- **Real MIMIC:** If you have MIMIC-IV access, place CSVs in `data/raw/` with the same table names; `preprocess.py` will work on them.
