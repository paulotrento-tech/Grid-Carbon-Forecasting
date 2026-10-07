"""
Day 2, step 2: exploratory analysis of California grid carbon intensity.

Usage:
    python src/eda.py

Input:  data/processed/caiso_clean.csv
Output: figures/*.png and a printed summary
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

IN = Path("data/processed/caiso_clean.csv")
FIG = Path("figures")

# Palette (validated categorical order + one-hue sequential ramp)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlecolor": TEXT, "axes.titlelocation": "left",
    "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "legend.frameon": False,
})

UNIT = "kg CO₂ / MWh"
SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"}


def load():
    df = pd.read_csv(IN, index_col="timestamp_utc", parse_dates=["timestamp_utc"])
    loc = df.index.tz_convert("America/Los_Angeles")
    df["date"] = loc.date
    df["hour"] = loc.hour
    df["month"] = loc.month
    df["year"] = loc.year
    df["season"] = df["month"].map(SEASONS)
    return df[df["year"] >= 2023]  # drop the few UTC hours that fall on 2022-12-31 locally


def fig_heatmap(df):
    grid = df.pivot_table(index="month", columns="hour", values="intensity", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", SEQ_BLUES)
    im = ax.imshow(grid.values, aspect="auto", cmap=cmap, origin="upper")
    ax.set_xticks(range(0, 24, 2), [f"{h}:00" for h in range(0, 24, 2)])
    ax.set_yticks(range(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.grid(False)
    ax.set_xlabel("Hour of day (local, hour start)")
    ax.set_title("Carbon intensity by month and hour: the clean window is widest in spring")
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label(UNIT, color=TEXT2)
    cb.outline.set_visible(False)
    fig.savefig(FIG / "1_heatmap_month_hour.png")
    plt.close(fig)


def fig_season_profiles(df):
    prof = df.pivot_table(index="hour", columns="season", values="intensity", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, s in enumerate(["Winter", "Spring", "Summer", "Fall"]):
        ax.plot(prof.index, prof[s], color=SERIES[i], label=s)
    ax.set_xticks(range(0, 24, 3), [f"{h}:00" for h in range(0, 24, 3)])
    ax.set_xlim(0, 23)
    ax.set_ylim(0, None)
    ax.set_ylabel(UNIT)
    ax.set_xlabel("Hour of day (local, hour start)")
    ax.set_title("Average daily profile by season")
    ax.legend(loc="lower left", ncol=4)
    fig.savefig(FIG / "2_daily_profile_by_season.png")
    plt.close(fig)


def fig_fuel_mix(df):
    shares = df.filter(like="share_").copy()
    shares.columns = [c.replace("share_", "") for c in shares.columns]
    groups = {
        "Natural gas": ["NG"], "Solar": ["SUN"], "Hydro": ["WAT"], "Wind": ["WND"],
        "Nuclear": ["NUC"], "Other": ["OTH", "COL", "OIL"],
    }
    mix = pd.DataFrame({k: shares[[c for c in v if c in shares]].sum(axis=1) for k, v in groups.items()})
    mix["hour"] = df["hour"]
    prof = mix.groupby("hour").mean() * 100

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.grid(False)
    ax.stackplot(prof.index, prof.T.values, labels=prof.columns,
                 colors=SERIES[:len(prof.columns)], edgecolor="white", linewidth=1)
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(0, 24, 3), [f"{h}:00" for h in range(0, 24, 3)])
    ax.set_ylabel("Share of in-state generation (%)")
    ax.set_xlabel("Hour of day (local, hour start)")
    ax.set_title("Solar displaces gas at midday; gas returns every evening")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.savefig(FIG / "3_fuel_mix_by_hour.png")
    plt.close(fig)


def fig_trend(df):
    daily = df.groupby(pd.to_datetime(df["date"]))["intensity"].mean()
    roll = daily.rolling(30, center=True, min_periods=15).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily.index, daily, color="#b9b8b2", linewidth=0.8, label="Daily mean")
    ax.plot(roll.index, roll, color=SERIES[0], label="30-day average")
    ax.set_ylabel(UNIT)
    ax.set_ylim(0, None)
    ax.set_title("Daily mean intensity: strong seasonality and a downward drift year over year")
    ax.legend(loc="lower left", ncol=2)
    fig.savefig(FIG / "4_daily_trend.png")
    plt.close(fig)


def fig_acf(df, max_lag=24 * 14):
    y = df["intensity"] - df["intensity"].mean()
    denom = (y * y).sum()
    acf = np.array([(y * y.shift(k)).sum() / denom for k in range(max_lag + 1)])
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(range(max_lag + 1), acf, color=SERIES[0], linewidth=1.5)
    for k, lbl in [(24, "1 day"), (168, "1 week")]:
        ax.axvline(k, color=TEXT2, linewidth=0.8, linestyle="--")
        ax.annotate(f"{lbl}: {acf[k]:.2f}", (k, acf[k]), xytext=(6, 8),
                    textcoords="offset points", color=TEXT, fontsize=9)
    ax.set_xticks(range(0, max_lag + 1, 24), [str(d) for d in range(15)])
    ax.set_xlim(0, max_lag)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation peaks every 24 hours: yesterday is a strong predictor of today")
    fig.savefig(FIG / "5_autocorrelation.png")
    plt.close(fig)
    return acf


def summary(df, acf):
    print("Mean intensity by year (kg CO2/MWh):")
    print(df.groupby("year")["intensity"].mean().round(1).to_string())

    by_day = df.groupby("date")["intensity"]
    spread = (by_day.max() - by_day.min()) / by_day.max()
    print(f"\nAverage within-day spread (max-min)/max: {spread.mean():.0%}")

    by_hour = df.pivot_table(index="date", columns="hour", values="intensity", aggfunc="mean")
    cleanest = by_hour.idxmin(axis=1)
    in_window = cleanest.between(9, 15).mean()
    print(f"Days where the cleanest hour falls 9:00-15:00: {in_window:.0%}")
    print(f"Most common cleanest hour: {cleanest.mode().iloc[0]}:00")

    print(f"\nAutocorrelation at 24 h: {acf[24]:.2f}  |  168 h: {acf[168]:.2f}")
    print(f"Imputed hours: {int(df['imputed'].sum())}")


def main():
    FIG.mkdir(exist_ok=True)
    df = load()
    fig_heatmap(df)
    fig_season_profiles(df)
    fig_fuel_mix(df)
    fig_trend(df)
    acf = fig_acf(df)
    summary(df, acf)
    print(f"\nSaved 5 figures to {FIG}/")


if __name__ == "__main__":
    main()
