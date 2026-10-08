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

### Key finding: accuracy is not the same as usefulness

The 7-day average has a **worse MAE** than "same hour yesterday" (24.0 vs 21.4) but **captures more savings** (95.3% vs 94.9%) and has lower regret (3.7 vs 4.1 kg/MWh). Averaging over a week smooths out one-off unusual days, so it ranks the hours better even though its values are further off.

For scheduling, a forecast only needs to get the **order** of the hours right, not the exact values. So forecasting models in this project are judged on decision metrics (regret, savings captured), not just MAE. Also, a naive forecast already captures about 95% of the possible savings, so a better model has to prove its value on the hard days (weather shifts, seasonal transitions) where yesterday is a poor guide.

## Day 3: forecasting models

```bash
python src/forecast.py   # writes results/forecast_metrics.csv, results/forecasts_2025.csv, 2 figures
```

Two models, both refit at the start of every 2025 month on all earlier data (walk-forward, so no model ever sees the future):
- **Ridge**: linear regression on lagged values, yesterday's fuel shares, and calendar features.
- **LightGBM**: gradient-boosted trees on the same features. It predicts each hour as a deviation from yesterday's daily mean (the day's *shape*), then adds the level back. This improved every metric for LightGBM; for Ridge it changes nothing, as expected for a linear model.

Results (test year 2025; "hard days" = the 20% of days where "same hour yesterday" missed the most):

| Model | MAE, all days | MAE, hard days | Savings captured, all | Savings captured, hard |
|---|---|---|---|---|
| Same hour yesterday | 21.4 | 40.4 | 94.9% | 91.4% |
| 7-day average | 24.0 | 33.3 | 95.3% | **93.9%** |
| Ridge | 17.6 | 27.5 | **95.8%** | 93.4% |
| LightGBM | **16.6** | **24.6** | 95.8% | 93.8% |

### Key finding: better forecasts, nearly the same decisions

LightGBM cuts error by 22% overall and 39% on hard days, but captured savings only rise from 94.9% to 95.8%. On hard days the simple 7-day average is still as good as anything for picking hours. The models are much better at predicting *how dirty* tomorrow will be, but only slightly better at predicting *which hours* will be cleanest, and the second question is what drives the schedule.

The likely reason: hard days are driven by weather (a cloudy day, a wind ramp), and none of the features know tomorrow's weather. Adding weather forecasts is the most promising next step for forecasting. For this project, the bigger lever is on the optimization side: with only ~4% of savings left on the table by the forecast, flexibility (battery, longer plug-in windows) matters more than forecast accuracy.

Top LightGBM features: same hour yesterday (33% of gain), yesterday's solar share at that hour (14%), most recent observed hour (10%).

## Assumptions and limitations

- **Direct emission factors** per fuel (see `EMISSION_FACTORS` in `src/build_intensity.py`). Lifecycle emissions are ignored.
- **In-state generation only.** EIA-930 fuel mix covers generation inside the CAISO balancing area. CAISO also imports a large share of its power, which is not reflected here.
- **Average, not marginal, intensity.** Shifting load changes which plant ramps up at the margin, so marginal intensity is the more correct signal. Average is used here as a tractable proxy.
- **Storage charging** (negative generation) is excluded from the generation total.
- **Grid batteries are not reported** as a separate fuel in this data, so their evening discharge (zero emissions) is missing from the total. Evening intensity is therefore likely somewhat overstated. This affects the size of the midday-evening gap, not which hours are cleanest.
- **Rooftop solar is not included.** EIA-930 covers utility-scale generation only; behind-the-meter solar shows up as lower demand, not generation.

## Roadmap

- Day 3: forecasting models (SARIMA, LightGBM), scored on decision metrics as well as MAE
- Day 4: EV charging LP (cars only), then a second scenario with an on-site battery (state-of-charge constraint, round-trip efficiency), including emissions savings vs. battery size
- Day 5: backtest over 2025 (charge on arrival vs. forecast schedule vs. perfect foresight)
- Day 6: Streamlit dashboard
- Day 7: write-up and polish
