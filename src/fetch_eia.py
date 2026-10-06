"""
Day 1, step 1: pull hourly generation by fuel type for CAISO from the EIA-930 API (v2).

Usage:
    export EIA_API_KEY=your_key_here        # free key from https://www.eia.gov/opendata/
    python src/fetch_eia.py --start 2023-01-01 --end 2025-12-31

Output:
    data/raw/caiso_fuel_mix.csv  (columns: period_utc, fueltype, mwh)
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
PAGE_SIZE = 5000  # EIA's max rows per request
OUT = Path("data/raw/caiso_fuel_mix.csv")


def fetch_page(api_key, start, end, offset, respondent="CISO"):
    params = {
        "api_key": api_key,
        "frequency": "hourly",          # UTC hours
        "data[0]": "value",
        "facets[respondent][]": respondent,
        "start": f"{start}T00",
        "end": f"{end}T23",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": offset,
        "length": PAGE_SIZE,
    }
    for attempt in range(5):
        r = requests.get(URL, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()["response"]
        time.sleep(2 ** attempt)  # back off on rate limits / transient errors
    r.raise_for_status()


def fetch_all(api_key, start, end):
    rows, offset = [], 0
    while True:
        resp = fetch_page(api_key, start, end, offset)
        batch = resp["data"]
        rows.extend(batch)
        total = int(resp["total"])
        offset += len(batch)
        print(f"  fetched {offset:,} / {total:,} rows")
        if not batch or offset >= total:
            break
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2025-12-31")
    args = p.parse_args()

    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        raise SystemExit("Set EIA_API_KEY first (free at https://www.eia.gov/opendata/).")

    print(f"Pulling CAISO fuel mix {args.start} to {args.end} ...")
    df = fetch_all(api_key, args.start, args.end)

    df = df.rename(columns={"period": "period_utc", "value": "mwh"})
    df["mwh"] = pd.to_numeric(df["mwh"], errors="coerce")
    df = df[["period_utc", "fueltype", "mwh"]]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} rows to {OUT}")
    print("Fuel types found:", sorted(df["fueltype"].unique()))


if __name__ == "__main__":
    main()
