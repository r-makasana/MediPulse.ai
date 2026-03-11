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
  - `stays_merged.csv` – one row per ICU stay with vitals, labs, and labels (for modeling)

## Schema

- **Clinical schema:** `data/clinical_schema.py` (chart/lab itemids, ranges, admission types, ICD samples).  
- **Real MIMIC:** If you have MIMIC-IV access, place CSVs in `data/raw/` with the same table names; `preprocess.py` will work on them.
