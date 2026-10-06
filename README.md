# Grid Carbon Forecasting + EV Charging Optimization

Forecast California's hourly grid carbon intensity, then schedule EV charging to minimize emissions.

## Setup

```bash
pip install -r requirements.txt
export EIA_API_KEY=your_key_here   # free at https://www.eia.gov/opendata/
```

## Day 1: data

```bash
python src/fetch_eia.py --start 2023-01-01 --end 2025-12-31
python src/build_intensity.py
```

Produces `data/processed/caiso_carbon_intensity.csv`: one row per hour with total generation, CO2, carbon intensity (kg CO2/MWh), and fuel shares.

## Assumptions and limitations

- **Direct emission factors** per fuel (see `EMISSION_FACTORS` in `src/build_intensity.py`). Lifecycle emissions are ignored.
- **In-state generation only.** EIA-930 fuel mix covers generation inside the CAISO balancing area. CAISO also imports a large share of its power, which is not reflected here.
- **Average, not marginal, intensity.** Shifting load changes which plant ramps up at the margin, so marginal intensity is the more correct signal. Average is used here as a tractable proxy.
- **Storage charging** (negative generation) is excluded from the generation total.
