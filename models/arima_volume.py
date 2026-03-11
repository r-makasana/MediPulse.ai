"""
ARIMA model for daily admission/ICU volume forecasting.
Fit: python -m models.arima_volume --data-dir data --horizon 14
"""

import argparse
from pathlib import Path

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR.parent / "data"
DEFAULT_RAW = DEFAULT_DATA / "raw"
DEFAULT_CKPT = SCRIPT_DIR / "checkpoints"


def load_volume_series(raw_dir: Path, series: str = "admissions") -> pd.Series:
    """Load admissions or icustays and aggregate to daily counts. Returns daily count series."""
    if series == "admissions":
        path = raw_dir / "admissions.csv"
    else:
        path = raw_dir / "icustays.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run data generation first.")
    df = pd.read_csv(path)
    if series == "admissions":
        df["date"] = pd.to_datetime(df["admittime"], format="mixed", errors="coerce").dt.date
    else:
        df["date"] = pd.to_datetime(df["intime"], format="mixed", errors="coerce").dt.date
    daily = df.groupby("date").size()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    return daily


def suggest_difference(series: pd.Series) -> int:
    """Suggest d for ARIMA: run ADF test; if not stationary, difference once."""
    adf = adfuller(series.dropna())
    if adf[1] < 0.05:
        return 0
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--series", choices=("admissions", "icustays"), default="admissions")
    parser.add_argument("--order", type=str, default=None, help="ARIMA(p,d,q) e.g. 2,1,1")
    parser.add_argument("--horizon", type=int, default=14, help="Forecast days ahead")
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_CKPT)
    args = parser.parse_args()
    args.raw_dir = args.raw_dir or args.data_dir / "raw"

    print(f"Loading daily {args.series} counts...")
    daily = load_volume_series(args.raw_dir, series=args.series)
    print(f"  Days: {len(daily)}, mean daily count: {daily.mean():.1f}")

    if args.order:
        p, d, q = map(int, args.order.split(","))
    else:
        d = suggest_difference(daily)
        # Simple default: (2, d, 1) or (1, d, 1)
        p, q = 2, 1
    print(f"Fitting ARIMA({p},{d},{q})...")
    model = ARIMA(daily.astype(float), order=(p, d, q))
    fitted = model.fit()
    print(fitted.summary().tables[1])

    forecast = fitted.forecast(steps=args.horizon)
    print(f"\nForecast next {args.horizon} days:")
    for i, val in enumerate(forecast.values, 1):
        print(f"  Day {i}:  {val:.1f}")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    # Save model (pickle) and forecast CSV
    fitted.save(str(args.save_dir / "arima_volume.pkl"))
    pd.DataFrame({"step": range(1, len(forecast) + 1), "forecast": forecast.values}).to_csv(
        args.save_dir / "arima_volume_forecast.csv", index=False
    )
    print(f"\nSaved model to {args.save_dir / 'arima_volume.pkl'}, forecast to arima_volume_forecast.csv")


if __name__ == "__main__":
    main()
