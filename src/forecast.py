"""
Day 3: day-ahead forecasting models for hourly carbon intensity.

Usage:
    python src/forecast.py

Input:  data/processed/caiso_clean.csv
Output: results/forecast_metrics.csv     all models (incl. baselines), all days and hard days
        results/forecasts_2025.csv       hourly forecasts for 2025 (used by the Day 5 backtest)
        figures/6_forecast_hard_week.png
        figures/7_feature_importance.png

Setup (same as baselines.py, so results are directly comparable):
    At local midnight before day d, forecast all 24 hours of day d using only data
    through the end of day d-1. Test year: 2025.

Walk-forward training:
    The models are refit at the start of every month of 2025 on ALL data before that
    month (expanding window). This mimics real use: you never train on the future.

Models:
    ridge   linear regression on lags + calendar features (interpretable)
    lgbm    LightGBM gradient-boosted trees on the same features (captures interactions,
            e.g. "yesterday's solar share matters more at noon than at midnight")

Hard days:
    The 20% of test days where the "yesterday" baseline missed the most. These are the
    days where a model has to add value beyond copying yesterday.
"""

from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from baselines import TEST_YEAR, daily_matrix, decision_metrics, make_forecasts, point_metrics

IN = Path("data/processed/caiso_clean.csv")
RESULTS = Path("results")
FIG = Path("figures")
HARD_DAY_SHARE = 0.20

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------- features

def build_features(df):
    """One row per (date, hour). Every feature uses only data from day d-1 or earlier."""
    Y = daily_matrix(df, "intensity")
    SUN = daily_matrix(df, "share_SUN")
    NG = daily_matrix(df, "share_NG")

    def per_hour(M, name):
        return M.stack(future_stack=True).rename(name)

    def per_day(s, name):
        return s.reindex(Y.index).rename(name)

    y1 = Y.shift(1)
    parts = [
        per_hour(Y, "target"),
        per_hour(y1, "lag1"),                                   # same hour yesterday
        per_hour(Y.shift(2), "lag2"),
        per_hour(Y.shift(7), "lag7"),                           # same hour last week
        per_hour(Y.rolling(7, min_periods=5).mean().shift(1), "mean7"),
        per_hour(Y.rolling(7, min_periods=5).std().shift(1), "std7"),
        per_hour(SUN.shift(1), "sun_share_lag1"),
        per_hour(NG.shift(1), "gas_share_lag1"),
    ]
    feats = pd.concat(parts, axis=1)

    day_level = pd.concat([
        per_day(y1.mean(axis=1), "lag1_day_mean"),
        per_day(y1.min(axis=1), "lag1_day_min"),
        per_day(y1.max(axis=1), "lag1_day_max"),
        per_day(y1[23], "last_obs"),                            # most recent hour we know
        per_day(SUN.shift(1).mean(axis=1), "lag1_day_sun_share"),
        per_day(Y.mean(axis=1).rolling(7, min_periods=5).mean().shift(1), "mean7_day"),
    ], axis=1)
    day_vals = day_level.reindex(feats.index.get_level_values("date"))
    day_vals.index = feats.index
    feats = feats.join(day_vals)

    dates = feats.index.get_level_values("date")
    feats["hour"] = feats.index.get_level_values("hour")
    feats["dow"] = dates.dayofweek
    feats["month"] = dates.month
    doy = dates.dayofyear
    feats["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    feats["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    feats["trend"] = (dates - dates.min()).days

    imputed = daily_matrix(df, "imputed").fillna(1).stack(future_stack=True) > 0
    feats["imputed"] = imputed.reindex(feats.index).fillna(True)
    return feats, Y


FEATURES = ["lag1", "lag2", "lag7", "mean7", "std7", "sun_share_lag1", "gas_share_lag1",
            "lag1_day_mean", "lag1_day_min", "lag1_day_max", "last_obs", "lag1_day_sun_share",
            "mean7_day", "hour", "dow", "month", "doy_sin", "doy_cos", "trend"]
CATEGORICAL = ["hour", "dow"]


# ---------------------------------------------------------------- models

def make_ridge():
    numeric = [f for f in FEATURES if f not in CATEGORICAL + ["month"]]
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return make_pipeline(pre, Ridge(alpha=1.0))


def make_lgbm():
    return lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        objective="l1",  # optimize absolute error directly; robust to outlier hours
        random_state=0, verbose=-1,
    )


MODELS = {"ridge": make_ridge, "lgbm": make_lgbm}

# Level features get re-expressed relative to yesterday's daily mean (see make_relative).
LEVEL_FEATURES = ["lag1", "lag2", "lag7", "mean7", "last_obs",
                  "lag1_day_min", "lag1_day_max", "mean7_day"]


def make_relative(feats):
    """
    Re-express target and level features as deviations from yesterday's daily mean.

    The trees then learn the SHAPE of tomorrow (which hours are above/below the day's
    level) separately from the level itself. Scheduling only cares about the shape, and
    trees can't extrapolate levels they haven't seen, so this helps on both counts.
    (A linear model gives identical results either way, which is a nice sanity check.)
    """
    rel = feats.copy()
    base = rel["lag1_day_mean"]
    for col in ["target"] + LEVEL_FEATURES:
        rel[col] = rel[col] - base
    return rel, base


def walk_forward(feats):
    """Refit each model at the start of every test month on all earlier data."""
    feats, base = make_relative(feats)
    usable = feats.dropna(subset=FEATURES)
    dates = usable.index.get_level_values("date")
    preds = {name: pd.Series(np.nan, index=feats.index) for name in MODELS}
    last_lgbm = None

    for month in range(1, 13):
        start = pd.Timestamp(TEST_YEAR, month, 1)
        end = start + pd.offsets.MonthBegin(1)
        train = usable[(dates < start) & ~usable["imputed"] & usable["target"].notna()]
        test = usable[(dates >= start) & (dates < end)]
        if test.empty:
            continue
        for name, make in MODELS.items():
            model = make().fit(train[FEATURES], train["target"])
            preds[name].loc[test.index] = model.predict(test[FEATURES])
            if name == "lgbm":
                last_lgbm = model
        print(f"  {start:%b %Y}: trained on {len(train):,} hours")
    preds = {name: p + base for name, p in preds.items()}  # back to absolute kg/MWh
    return preds, last_lgbm


# ---------------------------------------------------------------- evaluation

def to_matrix(series):
    return series.unstack("hour")


def evaluate(Y, forecasts, imputed_mask, days=None, label="all"):
    test = Y.index.year == TEST_YEAR
    y = Y[test]
    mask = (~imputed_mask[test]) & y.notna()
    if days is not None:
        keep = y.index.isin(days)
        y, mask = y[keep], mask[keep]
    rows = []
    for name, F in forecasts.items():
        f = F.reindex(index=y.index, columns=y.columns)
        m = mask & f.notna()
        rows.append({"model": name, "days": label,
                     **point_metrics(y, f, m), **decision_metrics(y, f, m)})
    return pd.DataFrame(rows)


def hard_days(Y, yesterday, imputed_mask):
    test = Y.index.year == TEST_YEAR
    err = (yesterday[test] - Y[test]).abs().where(~imputed_mask[test]).mean(axis=1)
    cutoff = err.quantile(1 - HARD_DAY_SHARE)
    return err[err >= cutoff].index


# ---------------------------------------------------------------- figures

def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT2)


def fig_hard_week(Y, forecasts, hard):
    # pick the 7-day window in the test year containing the most hard days
    test_days = Y.index[Y.index.year == TEST_YEAR]
    counts = pd.Series(test_days.isin(hard).astype(int), index=test_days).rolling(7).sum()
    end = counts.idxmax()
    days = pd.date_range(end - pd.Timedelta(days=6), end)

    def flat(M):
        s = M.reindex(index=days).stack(future_stack=True)
        s.index = [d + pd.Timedelta(hours=h) for d, h in s.index]
        return s

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(flat(Y).index, flat(Y).values, color=TEXT, linewidth=2, label="Actual")
    for i, name in enumerate(["yesterday", "lgbm"]):
        s = flat(forecasts[name])
        ax.plot(s.index, s.values, color=SERIES[i], linewidth=1.6,
                linestyle="--" if name == "yesterday" else "-",
                label={"yesterday": "Same hour yesterday", "lgbm": "LightGBM"}[name])
    style_axes(ax)
    ax.set_ylim(0, None)
    ax.set_ylabel("kg CO₂ / MWh", color=TEXT2)
    ax.set_title(f"Forecasts on a hard week ({days[0]:%b %d} to {days[-1]:%b %d, %Y})",
                 loc="left", fontweight="bold", color=TEXT)
    ax.legend(frameon=False, loc="lower left", ncol=3)
    fig.savefig(FIG / "6_forecast_hard_week.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_importance(model):
    imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES)
    imp = (imp / imp.sum() * 100).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(imp.index, imp.values, color=SERIES[0], height=0.6)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of total split gain (%)", color=TEXT2)
    ax.set_title("What LightGBM relies on (top 12 features)", loc="left",
                 fontweight="bold", color=TEXT)
    for y, v in enumerate(imp.values):
        ax.text(v + 0.4, y, f"{v:.0f}%", va="center", color=TEXT2, fontsize=9)
    fig.savefig(FIG / "7_feature_importance.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    df = pd.read_csv(IN, index_col="timestamp_utc", parse_dates=["timestamp_utc"])
    feats, Y = build_features(df)
    imputed_mask = daily_matrix(df, "imputed").fillna(1) > 0

    print("Walk-forward training (refit monthly):")
    preds, last_lgbm = walk_forward(feats)

    baselines = make_forecasts(Y)
    forecasts = {
        "yesterday": baselines["yesterday"],
        "avg_7day": baselines["avg_7day"],
        "ridge": to_matrix(preds["ridge"]),
        "lgbm": to_matrix(preds["lgbm"]),
    }

    hard = hard_days(Y, baselines["yesterday"], imputed_mask)
    res = pd.concat([
        evaluate(Y, forecasts, imputed_mask),
        evaluate(Y, forecasts, imputed_mask, days=hard, label="hard 20%"),
    ]).set_index(["days", "model"]).round(2)

    RESULTS.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    res.to_csv(RESULTS / "forecast_metrics.csv")

    test = Y.index.year == TEST_YEAR
    out = pd.DataFrame({"actual": Y[test].stack(future_stack=True)})
    for name, F in forecasts.items():
        out[name] = F[test].stack(future_stack=True)
    out.to_csv(RESULTS / "forecasts_2025.csv")

    fig_hard_week(Y, forecasts, hard)
    fig_importance(last_lgbm)

    pd.set_option("display.width", 140)
    print(f"\nDay-ahead forecasts, test year {TEST_YEAR}:\n")
    print(res.to_string())
    print(f"\nSaved {RESULTS}/forecast_metrics.csv, {RESULTS}/forecasts_2025.csv, and 2 figures.")


if __name__ == "__main__":
    main()
