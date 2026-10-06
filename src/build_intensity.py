"""
Day 1, step 2: turn the raw fuel mix into an hourly carbon intensity series.

Usage:
    python src/build_intensity.py

Input:
    data/raw/caiso_fuel_mix.csv
Output:
    data/processed/caiso_carbon_intensity.csv
        timestamp_utc, timestamp_local, total_mwh, co2_kg, intensity_kg_per_mwh,
        plus one share_<fuel> column per fuel type

Method:
    intensity_t = sum_f (gen_{f,t} * EF_f) / sum_f gen_{f,t}

    EF_f is a direct (smokestack) emission factor in kg CO2 per MWh.
    These are approximate and are the main modeling assumption here;
    document and cite whatever values you end up using.
"""

from pathlib import Path

import pandas as pd

RAW = Path("data/raw/caiso_fuel_mix.csv")
OUT = Path("data/processed/caiso_carbon_intensity.csv")
LOCAL_TZ = "America/Los_Angeles"

# kg CO2 per MWh, direct combustion only (no lifecycle emissions).
# Ballpark values in line with EPA eGRID / EIA fleet averages. Adjust and cite.
EMISSION_FACTORS = {
    "COL": 1000.0,  # coal
    "NG": 410.0,    # natural gas (CA fleet is mostly efficient combined cycle)
    "OIL": 800.0,   # petroleum
    "OTH": 300.0,   # other (biomass, waste, etc.); rough placeholder
    "NUC": 0.0,
    "SUN": 0.0,
    "WND": 0.0,
    "WAT": 0.0,
    "GEO": 0.0,
    "BAT": 0.0,     # storage discharge: emissions were counted when it charged
    "PS": 0.0,      # pumped storage, same logic
}


def load_raw(path=RAW):
    df = pd.read_csv(path)
    df["timestamp_utc"] = pd.to_datetime(df["period_utc"], format="%Y-%m-%dT%H", utc=True)
    df["mwh"] = pd.to_numeric(df["mwh"], errors="coerce")
    return df


def build(df):
    unknown = sorted(set(df["fueltype"]) - set(EMISSION_FACTORS))
    if unknown:
        print(f"WARNING: no emission factor for {unknown}; treating as 0. Add them to EMISSION_FACTORS.")

    # Storage shows up as negative generation while charging. Charging is load,
    # not generation, so drop negatives from the generation total.
    df["mwh"] = df["mwh"].clip(lower=0)

    wide = df.pivot_table(index="timestamp_utc", columns="fueltype", values="mwh", aggfunc="sum")

    # Make the hourly index complete so gaps are visible instead of silently skipped.
    full_index = pd.date_range(wide.index.min(), wide.index.max(), freq="h", tz="UTC")
    wide = wide.reindex(full_index)
    wide.index.name = "timestamp_utc"

    factors = pd.Series({f: EMISSION_FACTORS.get(f, 0.0) for f in wide.columns})
    total = wide.sum(axis=1, min_count=1)
    co2 = (wide.fillna(0) * factors).sum(axis=1)

    out = pd.DataFrame({
        "total_mwh": total,
        "co2_kg": co2.where(total.notna()),
    })
    out["intensity_kg_per_mwh"] = out["co2_kg"] / out["total_mwh"]

    shares = wide.div(total, axis=0).add_prefix("share_")
    out = out.join(shares)
    out.insert(0, "timestamp_local", out.index.tz_convert(LOCAL_TZ))
    return out


def report(out):
    n_missing = out["intensity_kg_per_mwh"].isna().sum()
    print(f"Hours: {len(out):,}  |  missing: {n_missing:,} ({n_missing / len(out):.1%})")
    print(out["intensity_kg_per_mwh"].describe().round(1).to_string())

    hourly = (out.assign(hour=out["timestamp_local"].dt.hour)
                 .groupby("hour")["intensity_kg_per_mwh"].mean().round(0))
    print("\nMean intensity by local hour (kg/MWh):")
    print(hourly.to_string())
    print(f"\nCleanest hour: {hourly.idxmin()}:00  |  Dirtiest hour: {hourly.idxmax()}:00")


def main():
    out = build(load_raw())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT)
    print(f"Saved {len(out):,} hourly rows to {OUT}\n")
    report(out)


if __name__ == "__main__":
    main()
