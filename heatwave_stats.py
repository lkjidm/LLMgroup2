"""Historical heatwave frequency tool for Uppsala (SMHI 1985–2026)."""
import pandas as pd
import numpy as np
from pathlib import Path

_CSV = Path("uppsala_tmax_1985_2026.csv")
_df = None

def _load():
    global _df
    if _df is None:
        _df = pd.read_csv(_CSV, parse_dates=["date"])
    return _df

def find_heatwaves(df, threshold, min_duration=3):
    temp = df.copy()
    temp["is_hot"] = temp["tmax"] >= threshold
    temp["group"] = (temp["is_hot"] != temp["is_hot"].shift()).cumsum()
    temp.loc[~temp["is_hot"], "group"] = 0
    hw = (
        temp[temp["is_hot"]]
        .groupby("group")
        .agg(
            start=("date", "min"),
            end=("date", "max"),
            duration=("date", "count"),
            max_tmax=("tmax", "max"),
        )
        .reset_index(drop=True)
    )
    return hw[hw["duration"] >= min_duration].reset_index(drop=True)

def historical_heatwave_stats(days: int, temp: float) -> dict:
    """
    Return how often a pattern of `days` consecutive days
    with Tmax >= `temp` occurred in Uppsala 1985–2026.
    """
    if not isinstance(days, (int, np.integer)) or days < 1:
        return {"error": "days must be an integer >= 1"}
    if not isinstance(temp, (int, float, np.floating)) or temp < -40 or temp > 45:
        return {"error": "temperature must be a realistic °C value"}

    df = _load()
    years = (df["date"].max() - df["date"].min()).days / 365.25

    # Choose the matching definition
    if temp >= 30 and days >= 3:
        level, threshold, min_d = "level2", 30.0, 3
        label = "≥30°C for ≥3 days (extreme / SMHI yellow-warning style)"
    elif temp >= 33 and days >= 1:
        level, threshold, min_d = "extreme_single", 33.0, 1
        label = "≥33°C even for 1 day"
    elif temp >= 27 and days >= 3:
        level, threshold, min_d = "level1", 27.0, 3
        label = "≥27°C for ≥3 days (health-relevant)"
    elif temp >= 25 and days >= 5:
        level, threshold, min_d = "level3", 25.0, 5
        label = "≥25°C for ≥5 days (classic SMHI heatwave)"
    else:
        return {
            "level": "none",
            "label": "Below defined heatwave thresholds",
            "historical_count": 0,
            "years_covered": round(years, 1),
            "empirical_rate_per_year": 0.0,
            "longest_similar": None,
            "record_max_tmax": float(df["tmax"].max()),
        }

    hw = find_heatwaves(df, threshold=threshold, min_duration=min_d)
    count = len(hw)
    rate = count / years if years > 0 else 0.0

    longest = None
    if count > 0:
        row = hw.loc[hw["duration"].idxmax()]
        longest = {
            "start": str(row["start"].date()),
            "end": str(row["end"].date()),
            "duration": int(row["duration"]),
            "max_tmax": float(row["max_tmax"]),
        }

    return {
        "level": level,
        "label": label,
        "query_days": int(days),
        "query_temp": float(temp),
        "historical_count": count,
        "years_covered": round(years, 1),
        "empirical_rate_per_year": round(rate, 3),
        "longest_similar": longest,
        "record_max_tmax": float(df["tmax"].max()),
        "station": "Uppsala Aut (97510)",
        "period": "1985-06-01 to 2026-06-30",
    }
