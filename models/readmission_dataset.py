"""
Build readmission label (30-day) and LSTM sequences from chartevents per stay.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from data.clinical_schema import CHART_ITEMS

# Fixed order for sequence features: [HR, SBP, DBP, Temp, Resp, SpO2]
CHART_ITEM_ORDER = list(CHART_ITEMS.keys())
N_VITAL_FEATURES = len(CHART_ITEM_ORDER)


def load_readmission_inputs(processed_dir: Path, raw_dir: Path):
    """Load chartevents_cleaned, icustays, admissions with datetimes."""
    chart_path = processed_dir / "chartevents_cleaned.csv"
    if not chart_path.exists():
        raise FileNotFoundError(f"Run preprocess first. Missing {chart_path}")
    chart = pd.read_csv(chart_path)
    chart["charttime"] = pd.to_datetime(chart["charttime"], format="mixed", errors="coerce")

    icustays_path = raw_dir / "icustays.csv"
    admissions_path = raw_dir / "admissions.csv"
    if not icustays_path.exists() or not admissions_path.exists():
        raise FileNotFoundError("Need raw/icustays.csv and raw/admissions.csv")
    icustays = pd.read_csv(icustays_path)
    icustays["intime"] = pd.to_datetime(icustays["intime"], format="mixed", errors="coerce")
    icustays["outtime"] = pd.to_datetime(icustays["outtime"], format="mixed", errors="coerce")
    admissions = pd.read_csv(admissions_path)
    admissions["admittime"] = pd.to_datetime(admissions["admittime"], format="mixed", errors="coerce")
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], format="mixed", errors="coerce")
    return chart, icustays, admissions


def build_readmission_labels(admissions: pd.DataFrame, days: int = 30) -> pd.Series:
    """For each hadm_id, 1 if same subject readmitted within `days` after dischtime, else 0."""
    adm = admissions.sort_values(["subject_id", "admittime"]).copy()
    adm["dischtime"] = pd.to_datetime(adm["dischtime"])
    readmit = {}
    for subject_id, grp in adm.groupby("subject_id"):
        grp = grp.sort_values("admittime")
        for i, row in grp.iterrows():
            hadm_id = row["hadm_id"]
            disch = row["dischtime"]
            # Any later admission for this subject within 30 days?
            later = grp[
                (grp["admittime"] > disch)
                & (grp["admittime"] <= disch + pd.Timedelta(days=days))
            ]
            readmit[hadm_id] = 1 if len(later) > 0 else 0
    return pd.Series(readmit)


def build_sequences_per_stay(
    chart: pd.DataFrame,
    icustays: pd.DataFrame,
    seq_len: int,
    interval_minutes: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each stay, build a sequence of vitals (time steps x N_VITAL_FEATURES).
    Returns (X, lengths): X padded to seq_len, lengths = actual length per stay.
    """
    chart = chart.merge(
        icustays[["stay_id", "intime", "outtime"]],
        on="stay_id",
        how="inner",
    )
    chart = chart[(chart["charttime"] >= chart["intime"]) & (chart["charttime"] <= chart["outtime"])]
    # Round charttime to interval bins
    chart["bin"] = (chart["charttime"] - chart["intime"]).dt.total_seconds() // (interval_minutes * 60)
    # Pivot: (stay_id, bin) -> one row per (stay_id, time_bin) with columns for each itemid
    piv = chart.pivot_table(
        index=["stay_id", "bin"],
        columns="itemid",
        values="valuenum",
        aggfunc="mean",
    ).reset_index()
    # Fill missing itemids with NaN; we'll fill with forward fill then 0
    for c in CHART_ITEM_ORDER:
        if c not in piv.columns:
            piv[c] = np.nan
    piv = piv[["stay_id", "bin"] + CHART_ITEM_ORDER]
    # Normalize to 0-1 using schema ranges for stability
    lo_hi = {i: (CHART_ITEMS[i][2], CHART_ITEMS[i][3]) for i in CHART_ITEM_ORDER}
    for i in CHART_ITEM_ORDER:
        lo, hi = lo_hi[i]
        piv[i] = (piv[i].ffill().fillna(lo) - lo) / max(hi - lo, 1e-6)
        piv[i] = piv[i].clip(0, 1)
    # Build array per stay
    stay_ids = piv["stay_id"].unique()
    X_list = []
    len_list = []
    for stay_id in stay_ids:
        sub = piv[piv["stay_id"] == stay_id].sort_values("bin")
        vals = sub[CHART_ITEM_ORDER].values.astype(np.float32)
        L = len(vals)
        len_list.append(min(L, seq_len))
        if L >= seq_len:
            vals = vals[-seq_len:]
        else:
            pad = np.zeros((seq_len - L, N_VITAL_FEATURES), dtype=np.float32)
            vals = np.vstack([pad, vals])
        X_list.append(vals)
    X = np.stack(X_list, axis=0)
    lengths = np.array(len_list, dtype=np.int64)
    return X, lengths, stay_ids


def get_readmission_dataset(
    processed_dir: Path,
    raw_dir: Path,
    seq_len: int = 96,
    interval_minutes: int = 60,
    readmission_days: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns X (n_stays, seq_len, n_features), y (n_stays,), lengths (n_stays,), stay_ids.
    Only includes stays that have both a sequence and a readmission label (hadm_id in admissions).
    """
    chart, icustays, admissions = load_readmission_inputs(processed_dir, raw_dir)
    # Readmission label per hadm_id
    readmit_series = build_readmission_labels(admissions, days=readmission_days)
    # Sequences per stay
    X, lengths, stay_ids = build_sequences_per_stay(
        chart, icustays, seq_len=seq_len, interval_minutes=interval_minutes
    )
    # Map stay_id -> hadm_id
    stay_to_hadm = icustays.set_index("stay_id")["hadm_id"]
    hadm_per_stay = stay_to_hadm.reindex(stay_ids).values
    # Only keep stays that have hadm_id in admissions (and thus a readmission label)
    valid = np.array([h in readmit_series.index for h in hadm_per_stay])
    X = X[valid]
    lengths = lengths[valid]
    stay_ids = stay_ids[valid]
    hadm_per_stay = hadm_per_stay[valid]
    y = np.array([readmit_series[h] for h in hadm_per_stay], dtype=np.int64)
    return X, y, lengths, stay_ids
