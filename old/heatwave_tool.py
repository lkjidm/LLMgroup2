# %% [markdown]
# <a href="https://colab.research.google.com/github/lkjidm/LLMgroup2/blob/br-1/02_Heatwave_tool.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
# Clone the repository
!git clone https://github.com/lkjidm/LLMgroup2

# Move into the repo folder
%cd LLMgroup2

# Switch to your branch (replace with the real branch name)
!git checkout br-1

# Check that you are on the correct branch and see the files
!git branch
!ls -la

# %%
import pandas as pd
import numpy as np

# Load the cleaned temperature data
df_hist = pd.read_csv("uppsala_tmax_1985_2026.csv", parse_dates=["date"])

print("Data loaded successfully!")
print("Shape:", df_hist.shape)
print("Date range:", df_hist["date"].min().date(), "→", df_hist["date"].max().date())
print("\nFirst 5 rows:")
print(df_hist.head())

# %%
hw_27 = pd.read_csv("heatwaves_level1_27C.csv", parse_dates=["start", "end"])
hw_30 = pd.read_csv("heatwaves_level2_30C.csv", parse_dates=["start", "end"])
hw_25 = pd.read_csv("heatwaves_level3_25C.csv", parse_dates=["start", "end"])

print(f"Level 1 events loaded: {len(hw_27)}")
print(f"Level 2 events loaded: {len(hw_30)}")
print(f"Level 3 events loaded: {len(hw_25)}")

# %% [markdown]
# # 02 - Heatwave Risk Tool (Uppsala)
# 
# This notebook contains the **deterministic risk function** that the LLM agent will call.
# 
# ### Purpose
# Given a simple query such as “3 days at 31 °C”, the function returns:
# - Which heatwave level it matches (Level 1 / 2 / 3 / none)
# - How many similar events occurred in Uppsala 1985-2026
# - A simple empirical risk score
# - A short explanation grounded in Swedish definitions
# 
# ### Data source
# - Station: 97510 - Uppsala Aut
# - Daily maximum temperature 1985-06-01 → 2026-06-30
# - Files loaded from the repository

# %%
def find_heatwaves(df, threshold, min_duration=3):
    """
    Return heatwave periods where Tmax >= threshold
    for at least min_duration consecutive days.
    """
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
            mean_tmax=("tmax", "mean")
        )
        .reset_index(drop=True)
    )
    return hw[hw["duration"] >= min_duration].reset_index(drop=True)


def compute_heatwave_risk(days: int, temp: float, df_hist=df_hist):
    """
    Deterministic heat-risk estimator for Uppsala.

    Levels:
    - level2          : ≥30°C for ≥3 days
    - level1          : ≥27°C for ≥3 days
    - level3          : ≥25°C for ≥5 days
    - extreme_single  : ≥33°C even for 1 day
    - none            : below all thresholds
    """
    # Basic input validation
    if not isinstance(days, (int, np.integer)) or days < 1:
        return {
            "level": "invalid",
            "description": "Invalid input",
            "historical_count": 0,
            "risk_score": 0.0,
            "comment": "Number of days must be an integer ≥ 1."
        }
    if not isinstance(temp, (int, float, np.floating)) or temp < -40 or temp > 45:
        return {
            "level": "invalid",
            "description": "Invalid input",
            "historical_count": 0,
            "risk_score": 0.0,
            "comment": "Temperature must be a realistic value (°C)."
        }

    # ----- Level 2: Extreme multi-day -----
    if temp >= 30 and days >= 3:
        hist = find_heatwaves(df_hist, threshold=30.0, min_duration=3)
        count = len(hist)
        score = min(0.95, 0.60 + 0.025 * max(0, 15 - count))
        return {
            "level": "level2",
            "description": "Extreme heat (Level 2): ≥30°C for ≥3 days",
            "historical_count": count,
            "risk_score": round(score, 2),
            "comment": "Matches SMHI yellow-warning level. Rare but high health impact in Uppsala."
        }

    # ----- Extreme single day -----
    if temp >= 33 and days >= 1:
        extreme_days = (df_hist["tmax"] >= 33).sum()
        score = min(0.75, 0.45 + 0.02 * extreme_days)
        return {
            "level": "extreme_single",
            "description": "Extreme single day: ≥33°C (even for 1 day)",
            "historical_count": int(extreme_days),
            "risk_score": round(score, 2),
            "comment": "Very high temperature for Uppsala. Notable even as a single day."
        }

    # ----- Level 1: Health-relevant -----
    if temp >= 27 and days >= 3:
        hist = find_heatwaves(df_hist, threshold=27.0, min_duration=3)
        count = len(hist)
        score = min(0.80, 0.30 + 0.006 * count)
        return {
            "level": "level1",
            "description": "Health-relevant heat (Level 1): ≥27°C for ≥3 days",
            "historical_count": count,
            "risk_score": round(score, 2),
            "comment": "Temperature range where Swedish studies show clear increases in mortality risk."
        }

    # ----- Level 3: Classic SMHI -----
    if temp >= 25 and days >= 5:
        hist = find_heatwaves(df_hist, threshold=25.0, min_duration=5)
        count = len(hist)
        score = min(0.65, 0.18 + 0.004 * count)
        return {
            "level": "level3",
            "description": "Classic SMHI heatwave (Level 3): ≥25°C for ≥5 days",
            "historical_count": count,
            "risk_score": round(score, 2),
            "comment": "Official SMHI climatological definition of a heatwave."
        }

    # Below all thresholds
    return {
        "level": "none",
        "description": "Below defined heatwave thresholds",
        "historical_count": 0,
        "risk_score": 0.05,
        "comment": "Does not meet any of the defined heat criteria."
    }

# %%
test_cases = [
    (3, 28.0),   # Level 1
    (4, 31.0),   # Level 2
    (6, 26.0),   # Level 3
    (2, 29.0),   # too short
    (5, 24.0),   # too cool
    (8, 32.5),   # strong Level 2
    (1, 35.0),   # single extreme day
]

print("Testing compute_heatwave_risk:\n")
for days, temp in test_cases:
    result = compute_heatwave_risk(days, temp)
    print(f"Query: {days} days at {temp}°C")
    print(f"  Level          : {result['level']}")
    print(f"  Description    : {result['description']}")
    print(f"  Historical cnt : {result['historical_count']}")
    print(f"  Risk score     : {result['risk_score']}")
    print(f"  Comment        : {result['comment']}")
    print("-" * 60)

# %%
print("Testing the new extreme_single rule:\n")

tests = [
    (1, 33.0),
    (1, 34.5),
    (2, 33.5),
    (1, 32.0),   # should still be "none"
]

for days, temp in tests:
    r = compute_heatwave_risk(days, temp)
    print(f"{days} day(s) at {temp}°C → {r['level']} | score={r['risk_score']} | {r['comment']}")


