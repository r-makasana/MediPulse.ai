"""
Preprocess synthetic (or real) MIMIC-like CSVs: clean, normalize, and build
per-stay feature tables for modeling.
Outputs: data/processed/
Run: python -m data.preprocess [--raw-dir data/raw] [--out-dir data/processed]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data.clinical_schema import CHART_ITEMS, LAB_ITEMS

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW = SCRIPT_DIR / "raw"
DEFAULT_PROCESSED = SCRIPT_DIR / "processed"


def load_raw(raw_dir: Path):
    """Load CSV tables from raw_dir. Missing files return None."""
    out = {}
    for name in ("patients", "admissions", "icustays", "chartevents", "labevents", "diagnoses_icd"):
        p = raw_dir / f"{name}.csv"
        if p.exists():
            out[name] = pd.read_csv(p)
            # Parse ISO datetimes (may include microseconds)
            for col in ("admittime", "dischtime", "intime", "outtime", "charttime", "storetime"):
                if col in out[name].columns:
                    out[name][col] = pd.to_datetime(out[name][col], format="mixed", errors="coerce")
        else:
            out[name] = None
    return out


def clean_chartevents(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    # Keep only known itemids with numeric valuenum
    valid_ids = set(CHART_ITEMS.keys())
    df = df[df["itemid"].isin(valid_ids)].copy()
    df = df.dropna(subset=["valuenum"])
    # Clamp to schema ranges
    for itemid, (_, _, lo, hi) in CHART_ITEMS.items():
        mask = df["itemid"] == itemid
        df.loc[mask, "valuenum"] = df.loc[mask, "valuenum"].clip(lo, hi)
    return df


def clean_labevents(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    valid_ids = set(LAB_ITEMS.keys())
    df = df[df["itemid"].isin(valid_ids)].copy()
    df = df.dropna(subset=["valuenum"])
    for itemid, (_, _, lo, hi) in LAB_ITEMS.items():
        mask = df["itemid"] == itemid
        df.loc[mask, "valuenum"] = df.loc[mask, "valuenum"].clip(lo, hi)
    return df


def build_stay_vitals(chart: pd.DataFrame, icustays: pd.DataFrame) -> pd.DataFrame:
    """Aggregate chartevents to per-stay vitals (mean, min, max, std for each item)."""
    if chart is None or chart.empty or icustays is None or icustays.empty:
        return pd.DataFrame()
    chart = chart.merge(
        icustays[["stay_id", "intime", "outtime"]],
        on="stay_id",
        how="inner",
    )
    chart = chart[chart["charttime"] >= chart["intime"]]
    chart = chart[chart["charttime"] <= chart["outtime"]]
    name_map = {i: (label.replace(" ", "_").lower()[:20]) for i, (label, _, _, _) in CHART_ITEMS.items()}
    rows = []
    for stay_id, g in chart.groupby("stay_id"):
        row = {"stay_id": stay_id}
        for itemid, sub in g.groupby("itemid"):
            prefix = name_map.get(itemid, f"item_{itemid}")
            row[f"{prefix}_mean"] = sub["valuenum"].mean()
            row[f"{prefix}_min"] = sub["valuenum"].min()
            row[f"{prefix}_max"] = sub["valuenum"].max()
            row[f"{prefix}_std"] = sub["valuenum"].std()
        rows.append(row)
    return pd.DataFrame(rows)


def build_stay_labs(lab: pd.DataFrame, icustays: pd.DataFrame) -> pd.DataFrame:
    """Aggregate labevents to per-stay: first, mean, worst (min/max by lab) per item."""
    if lab is None or lab.empty or icustays is None or icustays.empty:
        return pd.DataFrame()
    # Map stay_id via hadm_id (one ICU stay per hadm in our synthetic data)
    stay_hadm = icustays[["stay_id", "hadm_id"]].drop_duplicates()
    lab = lab.merge(stay_hadm, on="hadm_id", how="inner")
    lab = lab.merge(icustays[["stay_id", "intime", "outtime"]], on="stay_id", how="inner")
    lab = lab[lab["charttime"] >= lab["intime"]]
    lab = lab[lab["charttime"] <= lab["outtime"]]
    name_map = {i: (label.replace(" ", "_").lower()[:16]) for i, (label, _, _, _) in LAB_ITEMS.items()}
    rows = []
    for stay_id, g in lab.groupby("stay_id"):
        row = {"stay_id": stay_id}
        for itemid, sub in g.groupby("itemid"):
            prefix = name_map.get(itemid, f"lab_{itemid}")
            row[f"{prefix}_mean"] = sub["valuenum"].mean()
            row[f"{prefix}_first"] = sub.sort_values("charttime")["valuenum"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def build_stay_labels(admissions: pd.DataFrame, icustays: pd.DataFrame) -> pd.DataFrame:
    """Per-stay labels: demographics, los, mortality (if dod)."""
    if admissions is None or icustays is None:
        return pd.DataFrame()
    stay_adm = icustays.merge(
        admissions[["hadm_id", "subject_id", "admission_type", "insurance", "race"]],
        on=["hadm_id", "subject_id"],
        how="left",
    )
    stay_adm["los_icu_hours"] = (
        (stay_adm["outtime"] - stay_adm["intime"]).dt.total_seconds() / 3600
    )
    return stay_adm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PROCESSED)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    raw = load_raw(args.raw_dir)
    if raw.get("patients") is None:
        print("No raw data found. Run: python -m data.generate_synthetic_mimic")
        return

    print("Cleaning chartevents...")
    chart = clean_chartevents(raw["chartevents"])
    print("Cleaning labevents...")
    lab = clean_labevents(raw["labevents"])

    # Save cleaned event-level data
    if chart is not None and not chart.empty:
        chart.to_csv(args.out_dir / "chartevents_cleaned.csv", index=False)
    if lab is not None and not lab.empty:
        lab.to_csv(args.out_dir / "labevents_cleaned.csv", index=False)

    icustays = raw["icustays"]
    admissions = raw["admissions"]

    print("Building per-stay vitals...")
    stay_vitals = build_stay_vitals(chart, icustays)
    if not stay_vitals.empty:
        stay_vitals.to_csv(args.out_dir / "stay_vitals.csv", index=False)

    print("Building per-stay labs...")
    stay_labs = build_stay_labs(lab, icustays)
    if not stay_labs.empty:
        stay_labs.to_csv(args.out_dir / "stay_labs.csv", index=False)

    print("Building stay labels...")
    stay_labels = build_stay_labels(admissions, icustays)
    if not stay_labels.empty:
        stay_labels.to_csv(args.out_dir / "stay_labels.csv", index=False)

    # One merged table for modeling: stay_id + vitals + labs + labels
    if not stay_vitals.empty and (not stay_labs.empty or not stay_labels.empty):
        merged = stay_vitals
        if not stay_labs.empty:
            merged = merged.merge(stay_labs, on="stay_id", how="left")
        if not stay_labels.empty:
            merged = merged.merge(
                stay_labels[["stay_id", "subject_id", "hadm_id", "los_icu_hours", "admission_type", "insurance", "race"]],
                on="stay_id",
                how="left",
            )
        merged.to_csv(args.out_dir / "stays_merged.csv", index=False)
        print(f"Saved stays_merged.csv with {len(merged)} rows, {len(merged.columns)} columns")

    # Feature engineering: age, diagnoses, prior visits, labs, LOS
    try:
        from data.feature_engineering import build_featured_dataset
        print("Engineering features...")
        featured = build_featured_dataset(args.out_dir, args.raw_dir)
        featured.to_csv(args.out_dir / "stays_featured.csv", index=False)
        print(f"Saved stays_featured.csv with {len(featured)} rows, {len(featured.columns)} columns")
    except Exception as e:
        print(f"Feature engineering skipped: {e}")

    print(f"Done. Processed output in {args.out_dir}")


if __name__ == "__main__":
    main()
