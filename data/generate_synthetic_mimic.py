"""
Generate synthetic MIMIC-like ICU data using Faker and clinical schema.
Outputs CSV files to data/raw/ compatible with MIMIC-IV table names.
Run: python -m data.generate_synthetic_mimic [--n-patients 500] [--seed 42]
"""

import argparse
import csv
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from faker import Faker

from data.clinical_schema import (
    ADMISSION_TYPES,
    CAREUNITS,
    CHART_ITEMS,
    GENDERS,
    ICD_SAMPLES,
    INSURANCE,
    LAB_ITEMS,
    RACE,
)

# Default output dir (project root / data / raw)
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "raw"


def _writable_path(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_patients(fake: Faker, n: int, seed: int) -> list[dict]:
    np.random.seed(seed)
    rows = []
    for i in range(n):
        gender = np.random.choice(GENDERS)
        anchor_year = fake.random_int(min=2008, max=2022)
        anchor_age = fake.random_int(min=18, max=89)
        # Optional: ~10% mortality for synthetic realism
        dod = None
        if np.random.random() < 0.1:
            dod = fake.date_between(
                start_date=f"-{anchor_age + 5}y",
                end_date="today",
            ).isoformat()
        rows.append({
            "subject_id": i + 1,
            "gender": gender,
            "anchor_age": anchor_age,
            "anchor_year": anchor_year,
            "dod": dod or "",
        })
    return rows


def generate_admissions(
    fake: Faker,
    patient_rows: list[dict],
    seed: int,
    max_admissions_per_patient: int = 3,
) -> list[dict]:
    np.random.seed(seed + 1)
    rows = []
    hadm_id = 1
    for p in patient_rows:
        subject_id = p["subject_id"]
        n_adm = np.random.randint(1, max_admissions_per_patient + 1)
        base_time = fake.date_time_between(
            start_date=f"-{p['anchor_age']}y",
            end_date="now",
        )
        for _ in range(n_adm):
            admittime = base_time
            los_days = max(0, np.random.lognormal(mean=1.2, sigma=1.5))
            dischtime = admittime + timedelta(days=min(los_days, 365))
            base_time = dischtime + timedelta(days=np.random.randint(1, 800))
            rows.append({
                "hadm_id": hadm_id,
                "subject_id": subject_id,
                "admittime": admittime.isoformat(),
                "dischtime": dischtime.isoformat(),
                "admission_type": np.random.choice(ADMISSION_TYPES, p=[0.7, 0.25, 0.03, 0.02]),
                "insurance": np.random.choice(INSURANCE),
                "marital_status": np.random.choice(("Single", "Married", "Divorced", "Widowed", "Unknown"), p=(0.2, 0.5, 0.1, 0.1, 0.1)),
                "race": np.random.choice(RACE),
            })
            hadm_id += 1
    return rows


def generate_icustays(
    fake: Faker,
    admission_rows: list[dict],
    seed: int,
    icu_prob: float = 0.6,
) -> list[dict]:
    np.random.seed(seed + 2)
    rows = []
    stay_id = 1
    for a in admission_rows:
        if np.random.random() > icu_prob:
            continue
        hadm_id = a["hadm_id"]
        subject_id = a["subject_id"]
        admittime = datetime.fromisoformat(a["admittime"])
        dischtime = datetime.fromisoformat(a["dischtime"])
        # ICU stay within admission
        los_hours = max(12, np.random.lognormal(mean=3, sigma=1.2) * 24)
        intime = admittime + timedelta(hours=np.random.uniform(0, 24))
        outtime = intime + timedelta(hours=min(los_hours, (dischtime - intime).total_seconds() / 3600))
        if outtime > dischtime:
            outtime = dischtime
        careunit = np.random.choice(CAREUNITS)
        rows.append({
            "stay_id": stay_id,
            "hadm_id": hadm_id,
            "subject_id": subject_id,
            "intime": intime.isoformat(),
            "outtime": outtime.isoformat(),
            "first_careunit": careunit,
            "last_careunit": careunit,
        })
        stay_id += 1
    return rows


def generate_chartevents(
    icustay_rows: list[dict],
    seed: int,
    interval_minutes: int = 60,
) -> list[dict]:
    np.random.seed(seed + 3)
    rows = []
    for stay in icustay_rows:
        stay_id = stay["stay_id"]
        hadm_id = stay["hadm_id"]
        subject_id = stay["subject_id"]
        intime = datetime.fromisoformat(stay["intime"])
        outtime = datetime.fromisoformat(stay["outtime"])
        charttime = intime
        while charttime < outtime:
            for itemid, (label, uom, lo, hi) in CHART_ITEMS.items():
                # Slight correlation: worse vitals sometimes together
                noise = np.random.normal(0, (hi - lo) * 0.05)
                center = (lo + hi) / 2
                valuenum = np.clip(center + noise, lo, hi)
                valuenum = round(valuenum, 2)
                rows.append({
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "stay_id": stay_id,
                    "charttime": charttime.isoformat(),
                    "storetime": charttime.isoformat(),
                    "itemid": itemid,
                    "value": str(valuenum),
                    "valuenum": valuenum,
                    "valueuom": uom,
                })
            charttime += timedelta(minutes=interval_minutes)
    return rows


def generate_labevents(
    fake: Faker,
    admission_rows: list[dict],
    seed: int,
    labs_per_admission: tuple[int, int] = (5, 40),
) -> list[dict]:
    np.random.seed(seed + 4)
    rows = []
    for a in admission_rows:
        hadm_id = a["hadm_id"]
        subject_id = a["subject_id"]
        admittime = datetime.fromisoformat(a["admittime"])
        dischtime = datetime.fromisoformat(a["dischtime"])
        n_labs = np.random.randint(*labs_per_admission)
        for _ in range(n_labs):
            charttime = fake.date_time_between(
                start_date=admittime,
                end_date=dischtime,
            )
            itemid = np.random.choice(list(LAB_ITEMS.keys()))
            _, uom, lo, hi = LAB_ITEMS[itemid]
            valuenum = round(np.random.uniform(lo, hi), 3)
            rows.append({
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "charttime": charttime.isoformat(),
                "itemid": itemid,
                "value": str(valuenum),
                "valuenum": valuenum,
                "valueuom": uom,
            })
    return rows


def generate_diagnoses_icd(
    admission_rows: list[dict],
    seed: int,
    diagnoses_per_admission: tuple[int, int] = (1, 15),
) -> list[dict]:
    np.random.seed(seed + 5)
    rows = []
    for a in admission_rows:
        hadm_id = a["hadm_id"]
        subject_id = a["subject_id"]
        n_dx = np.random.randint(*diagnoses_per_admission)
        used = set()
        for seq in range(1, n_dx + 1):
            icd = np.random.choice(ICD_SAMPLES)
            if icd in used:
                continue
            used.add(icd)
            rows.append({
                "hadm_id": hadm_id,
                "subject_id": subject_id,
                "seq_num": seq,
                "icd_code": icd,
                "icd_version": 10,
            })
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic MIMIC-like data")
    parser.add_argument("--n-patients", type=int, default=500, help="Number of patients")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out-dir", type=Path, default=RAW_DIR, help="Output directory for CSVs")
    args = parser.parse_args()
    fake = Faker()
    Faker.seed(args.seed)

    out = _writable_path(args.out_dir)

    print("Generating patients...")
    patients = generate_patients(fake, args.n_patients, args.seed)
    write_csv(out / "patients.csv", patients)
    print("Generating admissions...")
    admissions = generate_admissions(fake, patients, args.seed)
    write_csv(out / "admissions.csv", admissions)
    print("Generating icustays...")
    icustays = generate_icustays(fake, admissions, args.seed)
    write_csv(out / "icustays.csv", icustays)
    print("Generating chartevents (vitals)...")
    chartevents = generate_chartevents(icustays, args.seed)
    write_csv(out / "chartevents.csv", chartevents)
    print("Generating labevents...")
    labevents = generate_labevents(fake, admissions, args.seed)
    write_csv(out / "labevents.csv", labevents)
    print("Generating diagnoses_icd...")
    diagnoses = generate_diagnoses_icd(admissions, args.seed)
    write_csv(out / "diagnoses_icd.csv", diagnoses)

    print(f"Done. Output in {out}")
    print(f"  patients: {len(patients)}, admissions: {len(admissions)}, icustays: {len(icustays)}")
    print(f"  chartevents: {len(chartevents)}, labevents: {len(labevents)}, diagnoses: {len(diagnoses)}")


if __name__ == "__main__":
    main()
