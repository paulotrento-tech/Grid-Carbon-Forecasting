"""
Day 2, step 1: clean the hourly carbon intensity series so it is safe to model.

Usage:
    python src/clean_data.py

Input:
    data/processed/caiso_carbon_intensity.csv   (from build_intensity.py)
Output:
    data/processed/caiso_clean.csv
        timestamp_utc (hour START), timestamp_local, intensity, imputed, + fuel shares

What it fixes (each one found while exploring the raw data):
  1. Hour labels. EIA-930 uses the hour-ENDING convention: a row stamped 15:00 covers
     14:00-15:00. We relabel every row to the hour it STARTS, which is what a scheduler
     means by "charge at 14:00".
  2. Missing natural gas. When the NG series is missing but other fuels are reported
     (e.g. 2023-11-09/10), intensity collapses toward 0 because the dirtiest fuel
     vanished. Those hours are fake-clean, so we blank them.
  3. Zero-generation hours (total_mwh == 0) are reporting glitches. Blank them.
  4. Gaps are filled: short gaps (<= 3 h) by linear interpolation, longer gaps with
     the value from the same hour one week earlier. Every filled hour is flagged with
     imputed = 1 so evaluation can exclude it.
"""

from pathlib import Path

import pandas as pd

IN = Path("data/processed/caiso_carbon_intensity.csv")
OUT = Path("data/processed/caiso_clean.csv")
LOCAL_TZ = "America/Los_Angeles"
SHORT_GAP_HOURS = 3


def load(path=IN):
    df = pd.read_csv(path, index_col="timestamp_utc", parse_dates=["timestamp_utc"])
    return df.drop(columns=["timestamp_local"])


def clean(df):
    df = df.copy()

    # 1. hour-ending -> hour-start
    df.index = df.index - pd.Timedelta(hours=1)
    df.index.name = "timestamp_utc"

    y = df["intensity_kg_per_mwh"].copy()

    # 2. gas missing while other generation is reported -> fake-clean hour
    gas_missing = df["share_NG"].isna() & (df["total_mwh"] > 0)
    # 3. zero total generation
    zero_total = df["total_mwh"].fillna(0) <= 0
    bad = gas_missing | zero_total
    y[bad] = float("nan")

    n_bad_flagged = int(bad.sum())
    n_missing = int(y.isna().sum())

    # 4a. short gaps: linear interpolation, only for runs of <= SHORT_GAP_HOURS
    is_na = y.isna()
    run_id = (is_na != is_na.shift()).cumsum()
    run_len = is_na.groupby(run_id).transform("sum")
    short = is_na & (run_len <= SHORT_GAP_HOURS)
    interp = y.interpolate(method="time", limit_area="inside")
    y[short] = interp[short]

    # 4b. long gaps: same hour one week earlier (repeat until filled)
    for _ in range(4):
        if not y.isna().any():
            break
        y = y.fillna(y.shift(168))

    imputed = is_na & y.notna()

    out = pd.DataFrame({"intensity": y, "imputed": imputed.astype(int)})
    shares = df.filter(like="share_").drop(columns=["share_GEO"], errors="ignore")
    out = out.join(shares)
    out.insert(0, "timestamp_local", out.index.tz_convert(LOCAL_TZ))

    print(f"Flagged as bad data: {n_bad_flagged} hours (gas missing or zero generation)")
    print(f"Total missing before fill: {n_missing} hours")
    print(f"Imputed: {int(imputed.sum())}  |  still missing: {int(out['intensity'].isna().sum())}")
    return out


def main():
    out = clean(load())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT)
    print(f"Saved {len(out):,} rows to {OUT}")


if __name__ == "__main__":
    main()
