"""
Engineer features for modeling: age, diagnosis codes, prior visits, lab values, length of stay.
Reads from data/raw and data/processed; writes data/processed/stays_featured.csv.
Run: python -m data.feature_engineering [--processed-dir data/processed] [--out-dir data/processed]
"""

import argparse
from pathlib import Path

import pandas as pd

from data.clinical_schema import ICD_SAMPLES

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW = SCRIPT_DIR / "raw"
DEFAULT_PROCESSED = SCRIPT_DIR / "processed"

# Short names for diagnosis one-hot columns (order matches ICD_SAMPLES)
ICD_FEATURE_NAMES = [
    "dx_hypertension",
    "dx_diabetes",
    "dx_pneumonia",
    "dx_heart_failure",
    "dx_aki",
    "dx_gerd",
    "dx_copd",
    "dx_cad",
    "dx_dyspnea",
    "dx_sepsis",
    "dx_severe_sepsis",
]


def load_processed(processed_dir: Path) -> dict:
    """Load processed CSVs. Returns dict with keys like stays_merged, stay_labels, etc."""
    out = {}
    for name in ("stays_merged", "stay_labels", "stay_vitals", "stay_labs"):
        p = processed_dir / f"{name}.csv"
        if p.exists():
            out[name] = pd.read_csv(p)
            for col in ("intime", "outtime"):
                if col in out[name].columns:
                    out[name][col] = pd.to_datetime(out[name][col], format="mixed", errors="coerce")
        else:
            out[name] = None
    return out


def load_raw_for_features(raw_dir: Path) -> dict:
    """Load raw tables needed for feature engineering."""
    out = {}
    for name in ("patients", "admissions", "diagnoses_icd"):
        p = raw_dir / f"{name}.csv"
        if p.exists():
            out[name] = pd.read_csv(p)
            for col in ("admittime", "dischtime"):
                if col in out[name].columns:
                    out[name][col] = pd.to_datetime(out[name][col], format="mixed", errors="coerce")
        else:
            out[name] = None
    return out


def add_age_features(
    df: pd.DataFrame,
    patients: pd.DataFrame,
    admissions: pd.DataFrame,
    icustays: pd.DataFrame,
) -> pd.DataFrame:
    """Add age at admission and anchor_age. Requires stay_id -> subject_id, hadm_id."""
    if patients is None or admissions is None or icustays is None:
        return df
    # Stay-level: stay_id, hadm_id, subject_id. Get admittime from admissions.
    stay_adm = icustays[["stay_id", "hadm_id", "subject_id"]].merge(
        admissions[["hadm_id", "admittime"]],
        on="hadm_id",
        how="left",
    )
    stay_adm = stay_adm.merge(
        patients[["subject_id", "anchor_age", "anchor_year"]],
        on="subject_id",
        how="left",
    )
    # Age at admission = anchor_age + (admission_year - anchor_year)
    stay_adm["admission_year"] = stay_adm["admittime"].dt.year
    stay_adm["age_at_admission"] = (
        stay_adm["anchor_age"] + (stay_adm["admission_year"] - stay_adm["anchor_year"])
    ).clip(0, 120)
    stay_adm["anchor_age"] = stay_adm["anchor_age"]  # keep for reference
    df = df.merge(
        stay_adm[["stay_id", "age_at_admission", "anchor_age"]],
        on="stay_id",
        how="left",
    )
    return df


def add_diagnosis_features(
    df: pd.DataFrame,
    diagnoses_icd: pd.DataFrame,
    admissions: pd.DataFrame,
) -> pd.DataFrame:
    """Add diagnosis count per admission and one-hot for common ICD codes."""
    if diagnoses_icd is None or admissions is None:
        return df
    # Diagnoses are at hadm_id level. Stays have hadm_id.
    # Count diagnoses per hadm_id
    dx_count = (
        diagnoses_icd.groupby("hadm_id", as_index=False)
        .agg(n_diagnoses=("icd_code", "nunique"))
    )
    df = df.merge(dx_count, on="hadm_id", how="left")
    df["n_diagnoses"] = df["n_diagnoses"].fillna(0).astype(int)

    # One-hot for ICD_SAMPLES (order must match ICD_FEATURE_NAMES)
    for idx, icd in enumerate(ICD_SAMPLES):
        if idx >= len(ICD_FEATURE_NAMES):
            break
        col = ICD_FEATURE_NAMES[idx]
        hadms_with_icd = set(
            diagnoses_icd[diagnoses_icd["icd_code"] == icd]["hadm_id"].unique()
        )
        df[col] = df["hadm_id"].isin(hadms_with_icd).astype(int)
    return df


def add_prior_visit_features(
    df: pd.DataFrame,
    admissions: pd.DataFrame,
    icustays: pd.DataFrame,
) -> pd.DataFrame:
    """Add prior_admissions and prior_icu_visits (count of visits before this stay)."""
    if admissions is None or icustays is None:
        return df
    # For each admission: subject_id, admittime. Sort by subject_id, admittime.
    adm = admissions[["hadm_id", "subject_id", "admittime"]].sort_values(
        ["subject_id", "admittime"]
    )
    adm["prior_admissions"] = adm.groupby("subject_id").cumcount()  # 0, 1, 2, ...
    # prior_admissions count = current index; "prior" = prior_admissions (we want count of *previous* admissions)
    adm["prior_admissions"] = adm.groupby("subject_id")["prior_admissions"].shift(1).fillna(0).astype(int)

    # Prior ICU visits: for each stay, count prior icustays for same subject (by intime)
    stays = icustays[["stay_id", "hadm_id", "subject_id", "intime"]].copy()
    stays = stays.sort_values(["subject_id", "intime"])
    stays["prior_icu_visits"] = stays.groupby("subject_id").cumcount()
    stays["prior_icu_visits"] = stays.groupby("subject_id")["prior_icu_visits"].shift(1).fillna(0).astype(int)

    df = df.merge(adm[["hadm_id", "prior_admissions"]], on="hadm_id", how="left")
    df = df.merge(stays[["stay_id", "prior_icu_visits"]], on="stay_id", how="left")
    df["prior_admissions"] = df["prior_admissions"].fillna(0).astype(int)
    df["prior_icu_visits"] = df["prior_icu_visits"].fillna(0).astype(int)
    return df


def add_los_features(
    df: pd.DataFrame,
    admissions: pd.DataFrame,
    icustays: pd.DataFrame,
) -> pd.DataFrame:
    """Ensure LOS features: los_icu_hours, los_hospital_hours (from admission)."""
    if "los_icu_hours" not in df.columns and icustays is not None:
        icustays = icustays.copy()
        icustays["los_icu_hours"] = (
            (icustays["outtime"] - icustays["intime"]).dt.total_seconds() / 3600
        )
        df = df.merge(icustays[["stay_id", "los_icu_hours"]], on="stay_id", how="left")
    if admissions is not None:
        admissions = admissions.copy()
        admissions["los_hospital_hours"] = (
            (admissions["dischtime"] - admissions["admittime"]).dt.total_seconds() / 3600
        )
        df = df.merge(
            admissions[["hadm_id", "los_hospital_hours"]],
            on="hadm_id",
            how="left",
        )
    return df


def build_featured_dataset(
    processed_dir: Path,
    raw_dir: Path,
) -> pd.DataFrame:
    """Build one DataFrame with all engineered features."""
    proc = load_processed(processed_dir)
    raw = load_raw_for_features(raw_dir)
    merged = proc.get("stays_merged")
    if merged is None or merged.empty:
        raise FileNotFoundError(
            f"stays_merged.csv not found in {processed_dir}. Run: python -m data.preprocess"
        )
    # Load icustays for age and prior visits (need intime, outtime)
    icustays_path = raw_dir / "icustays.csv"
    if icustays_path.exists():
        icustays = pd.read_csv(icustays_path)
        icustays["intime"] = pd.to_datetime(icustays["intime"], format="mixed", errors="coerce")
        icustays["outtime"] = pd.to_datetime(icustays["outtime"], format="mixed", errors="coerce")
    else:
        icustays = None

    df = merged.copy()
    df = add_age_features(df, raw["patients"], raw["admissions"], icustays)
    df = add_diagnosis_features(df, raw["diagnoses_icd"], raw["admissions"])
    df = add_prior_visit_features(df, raw["admissions"], icustays)
    df = add_los_features(df, raw["admissions"], icustays)
    return df


def main():
    parser = argparse.ArgumentParser(description="Engineer features for ICU stays")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: same as --processed-dir")
    parser.add_argument("-o", "--output", type=str, default="stays_featured.csv", help="Output filename")
    args = parser.parse_args()
    args.out_dir = args.out_dir or args.processed_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Building featured dataset...")
    df = build_featured_dataset(args.processed_dir, args.raw_dir)
    out_path = args.out_dir / args.output
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows, {len(df.columns)} columns to {out_path}")
    # Summary of engineered columns
    feat_cols = [
        "age_at_admission", "anchor_age", "n_diagnoses",
        "prior_admissions", "prior_icu_visits",
        "los_icu_hours", "los_hospital_hours",
    ] + ICD_FEATURE_NAMES
    present = [c for c in feat_cols if c in df.columns]
    print("Engineered features:", ", ".join(present))


if __name__ == "__main__":
    main()
