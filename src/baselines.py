"""
Day 2, step 3: naive day-ahead forecasting baselines. Day 3's models have to beat these.

Usage:
    python src/baselines.py

Setup (day-ahead):
    At local midnight before day d, forecast all 24 hours of day d using only data
    through the end of day d-1. Train period: 2023-2024. Test period: 2025.
    Imputed hours are excluded from scoring.

Baselines:
    yesterday     same hour on day d-1
    last_week     same hour on day d-7
    avg_7day      mean of the same hour over days d-7 .. d-1
    climatology   mean of the same (month, hour) in the training years

Metrics:
    MAE, RMSE, MAPE              how close the numbers are
    decision metrics (k=4 hours) how good the forecast is for SCHEDULING:
        pick the 4 cleanest hours of the day according to the forecast, then measure
        - regret:   extra intensity vs. the truly cleanest 4 hours (oracle)
        - captured: share of the possible savings vs. charging 18:00-22:00 (on arrival)
                    that the forecast actually achieved (100% = perfect)
"""

from pathlib import Path

import numpy as np
import pandas as pd

IN = Path("data/processed/caiso_clean.csv")
OUT = Path("results/baselines.csv")
TEST_YEAR = 2025
K = 4
ARRIVAL_HOURS = [18, 19, 20, 21]


def daily_matrix(df, col):
    loc = df.index.tz_convert("America/Los_Angeles")
    m = df.assign(date=pd.to_datetime(loc.date), hour=loc.hour).pivot_table(
        index="date", columns="hour", values=col, aggfunc="mean")
    return m.asfreq("D")  # one row per calendar day, gaps become NaN rows


def make_forecasts(Y):
    train = Y[Y.index.year < TEST_YEAR]
    clim = train.groupby(train.index.month).mean()
    climatology = pd.DataFrame(clim.loc[Y.index.month].values, index=Y.index, columns=Y.columns)
    return {
        "yesterday": Y.shift(1),
        "last_week": Y.shift(7),
        "avg_7day": Y.rolling(7, min_periods=5).mean().shift(1),
        "climatology": climatology,
    }


def point_metrics(y, f, mask):
    e = (f - y)[mask].stack()
    a = y[mask].stack()
    return {
        "MAE": e.abs().mean(),
        "RMSE": np.sqrt((e ** 2).mean()),
        "MAPE_%": (e.abs() / a).mean() * 100,
    }


def decision_metrics(y, f, mask):
    full_days = mask.all(axis=1) & f.notna().all(axis=1)
    yv, fv = y[full_days].values, f[full_days].values
    chosen = np.take_along_axis(yv, np.argsort(fv, axis=1)[:, :K], axis=1).mean(axis=1)
    oracle = np.sort(yv, axis=1)[:, :K].mean(axis=1)
    arrival = y.loc[full_days, ARRIVAL_HOURS].values.mean(axis=1)
    return {
        "regret_kg_per_mwh": (chosen - oracle).mean(),
        "captured_savings_%": (arrival - chosen).sum() / (arrival - oracle).sum() * 100,
        "days_scored": int(full_days.sum()),
    }


def main():
    df = pd.read_csv(IN, index_col="timestamp_utc", parse_dates=["timestamp_utc"])
    Y = daily_matrix(df, "intensity")
    imputed = daily_matrix(df, "imputed").fillna(1) > 0

    test = Y.index.year == TEST_YEAR
    y = Y[test]
    mask = (~imputed[test]) & y.notna()

    rows = []
    for name, F in make_forecasts(Y).items():
        f = F[test]
        m = mask & f.notna()
        rows.append({"model": name, **point_metrics(y, f, m), **decision_metrics(y, f, m)})

    res = pd.DataFrame(rows).set_index("model").round(2).sort_values("MAE")
    OUT.parent.mkdir(exist_ok=True)
    res.to_csv(OUT)

    pd.set_option("display.width", 120)
    print(f"Day-ahead baselines, test year {TEST_YEAR}:\n")
    print(res.to_string())
    best = res.index[0]
    print(f"\nBest baseline by MAE: {best} ({res.loc[best, 'MAE']:.1f} kg/MWh). "
          f"This is the number Day 3 has to beat.")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
