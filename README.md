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

## Day 2: cleaning, EDA, baselines

```bash
python src/clean_data.py   # fixes hour labels, blanks bad hours, fills gaps
python src/eda.py          # writes figures/*.png
python src/baselines.py    # writes results/baselines.csv
```

Data issues found and fixed in `clean_data.py`:
- EIA-930 stamps each hour by its **end** (per EIA's documentation). Rows are relabeled to the hour start.
- On 2023-11-09/10 the natural gas series is missing while other fuels are reported, which makes intensity fall to ~3 kg/MWh. These "fake clean" hours are blanked.
- 49 bad or missing hours in total (including all of 2024-11-02) are filled and flagged `imputed = 1`, and excluded from scoring.

Baseline results (day-ahead, test year 2025):

| Baseline | MAE (kg/MWh) | MAPE | Savings captured (4 h EV charge) |
|---|---|---|---|
| Same hour yesterday | 21.4 | 12.6% | 94.9% |
| 7-day average | 24.0 | 14.4% | 95.3% |
| Same hour last week | 31.6 | 18.6% | 91.6% |
| Month-hour climatology | 32.7 | 21.6% | 84.3% |

"Savings captured" = share of the emissions reduction a perfect forecast would achieve (vs. charging 18:00 to 22:00) when the forecast is used to pick the 4 cleanest hours.

## Assumptions and limitations

- **Direct emission factors** per fuel (see `EMISSION_FACTORS` in `src/build_intensity.py`). Lifecycle emissions are ignored.
- **In-state generation only.** EIA-930 fuel mix covers generation inside the CAISO balancing area. CAISO also imports a large share of its power, which is not reflected here.
- **Average, not marginal, intensity.** Shifting load changes which plant ramps up at the margin, so marginal intensity is the more correct signal. Average is used here as a tractable proxy.
- **Storage charging** (negative generation) is excluded from the generation total.
