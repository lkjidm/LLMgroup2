# %% [markdown]
# <a href="https://colab.research.google.com/github/lkjidm/LLMgroup2/blob/br-1/P01_SMHI_data_EDA.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
!nvidia-smi

# %%
!pip install -q requests pandas numpy matplotlib seaborn plotly scipy statsmodels
# Later we will add more (google-genai, transformers, etc.) when we build the agent

# %%
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
from IPython.display import display, Markdown

# %%
from google.colab import userdata
# Optional – we only need these when we build the LLM agent
try:
    GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")
    print("Gemini key loaded")
except:
    GEMINI_API_KEY = None
    print("No Gemini key yet")

try:
    HF_TOKEN = userdata.get("HF_TOKEN")
    print("HF token loaded")
except:
    HF_TOKEN = None
    print("No HF token yet")

# %%
# Entry point of the Meteorological Observations API
url = "https://opendata-download-metobs.smhi.se/api/version/latest.json"
r = requests.get(url)
print("Status:", r.status_code)
data = r.json()

# Show available parameters (first 15)
print("\nAvailable parameters (sample):")
for resource in data.get("resource", [])[:15]:
    print(f"  ID {resource['key']}: {resource['title']} ({resource.get('summary', '')})")

# %%
# Look at stations for parameter 1 (hourly temperature)
param_url = "https://opendata-download-metobs.smhi.se/api/version/latest/parameter/1.json"
r = requests.get(param_url)
stations = r.json().get("station", [])

# Filter for Uppsala-related stations
uppsala_stations = [s for s in stations if "uppsala" in s.get("name", "").lower()]
print(f"Found {len(uppsala_stations)} Uppsala-related stations:\n")
for s in uppsala_stations:
    print(f"ID: {s['id']}  |  Name: {s['name']}  |  Active: {s.get('active')}  |  Lat/Lon: {s.get('latitude')}, {s.get('longitude')}")

# %%
# Example: corrected-archive for daily max temperature (parameter 20) at Uppsala Aut (97510)
# First check available periods
period_url = "https://opendata-download-metobs.smhi.se/api/version/latest/parameter/20/station/97510.json"
r = requests.get(period_url)
print("Available periods:")
for p in r.json().get("period", []):
    print(f"  {p['key']}: {p['title']}")

# %%
# Download CSV for a period (example – adjust the period key after you see the list above)
data_url = "https://opendata-download-metobs.smhi.se/api/version/latest/parameter/20/station/97510/period/latest-months/data.csv"
df = pd.read_csv(data_url, sep=";", skiprows=10)  # SMHI CSVs often have header rows
print(df.head(20))
print("\nColumns:", df.columns.tolist())
print("\nShape:", df.shape)

# %%
# Clean the data we just downloaded
df_clean = df.copy()

# Keep only rows that have a real date and a temperature value
df_clean = df_clean.dropna(subset=["Representativt dygn", "Lufttemperatur"])

# Convert date column
df_clean["date"] = pd.to_datetime(df_clean["Representativt dygn"])

# Keep only the useful columns and rename
df_clean = df_clean[["date", "Lufttemperatur", "Kvalitet"]].rename(
    columns={"Lufttemperatur": "tmax", "Kvalitet": "quality"}
)

# Sort by date
df_clean = df_clean.sort_values("date").reset_index(drop=True)

print("Cleaned data shape:", df_clean.shape)
print("\nFirst 10 rows:")
print(df_clean.head(10))
print("\nLast 5 rows:")
print(df_clean.tail())
print("\nBasic statistics:")
print(df_clean["tmax"].describe())

# %%
plt.figure(figsize=(12, 5))
plt.plot(df_clean["date"], df_clean["tmax"], marker="o", markersize=3)
plt.axhline(25, color="orange", linestyle="--", label="25 °C")
plt.axhline(30, color="red", linestyle="--", label="30 °C")
plt.title("Daily maximum temperature – Uppsala Aut (last ~4 months)")
plt.ylabel("Tmax (°C)")
plt.xlabel("Date")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Download quality-controlled historical daily max temperature (parameter 20, station 97510)
archive_url = "https://opendata-download-metobs.smhi.se/api/version/latest/parameter/20/station/97510/period/corrected-archive/data.csv"

# SMHI puts metadata at the top – we skip the first ~10 rows and let pandas handle the rest
df_hist = pd.read_csv(archive_url, sep=";", skiprows=10)

print("Raw archive shape:", df_hist.shape)
print("\nFirst 5 rows of raw data:")
print(df_hist.head())
print("\nColumns:", df_hist.columns.tolist())

# %%
# Clean exactly like before
df_hist_clean = df_hist.copy()
df_hist_clean = df_hist_clean.dropna(subset=["Representativt dygn", "Lufttemperatur"])
df_hist_clean["date"] = pd.to_datetime(df_hist_clean["Representativt dygn"])
df_hist_clean = df_hist_clean[["date", "Lufttemperatur", "Kvalitet"]].rename(
    columns={"Lufttemperatur": "tmax", "Kvalitet": "quality"}
)
df_hist_clean = df_hist_clean.sort_values("date").reset_index(drop=True)

print("Cleaned historical data shape:", df_hist_clean.shape)
print("\nDate range:")
print("  First day:", df_hist_clean["date"].min().date())
print("  Last day :", df_hist_clean["date"].max().date())
print("\nBasic statistics:")
print(df_hist_clean["tmax"].describe())
print("\nFirst 5 rows:")
print(df_hist_clean.head())
print("\nLast 5 rows:")
print(df_hist_clean.tail())

# %%
# Work on a copy of the historical data
df = df_hist_clean.copy()

# Define heat days (you can change the threshold later)
THRESHOLD = 30.0          # °C – classic Swedish heat threshold
df["is_heat_day"] = df["tmax"] >= THRESHOLD

# Identify consecutive heatwave periods
df["heatwave_id"] = (df["is_heat_day"] != df["is_heat_day"].shift()).cumsum()
df.loc[~df["is_heat_day"], "heatwave_id"] = 0

# Keep only real heatwaves (at least 1 day for now; we can require ≥2 or ≥3 later)
heatwaves = (
    df[df["is_heat_day"]]
    .groupby("heatwave_id")
    .agg(
        start=("date", "min"),
        end=("date", "max"),
        duration=("date", "count"),
        max_tmax=("tmax", "max"),
        mean_tmax=("tmax", "mean")
    )
    .reset_index(drop=True)
)

print(f"Number of heat days (Tmax ≥ {THRESHOLD}°C): {df['is_heat_day'].sum()}")
print(f"Number of heatwave periods: {len(heatwaves)}")
print("\nHeatwaves found:")
print(heatwaves.sort_values("start"))

# %%
print("Heatwave duration statistics:")
print(heatwaves["duration"].describe())

print("\nHottest heatwaves (by max temperature):")
print(heatwaves.sort_values("max_tmax", ascending=False).head(10))

# %%
plt.figure(figsize=(14, 5))
plt.scatter(df["date"], df["tmax"], c=df["is_heat_day"].map({True: "red", False: "steelblue"}), s=8, alpha=0.6)
plt.axhline(THRESHOLD, color="red", linestyle="--", label=f"{THRESHOLD} °C threshold")
plt.title(f"Daily maximum temperature – Uppsala Aut (1985–2026)\nRed points = heat days ≥ {THRESHOLD}°C")
plt.ylabel("Tmax (°C)")
plt.xlabel("Date")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Progress so far – SMHI Data & Heatwave Analysis (Uppsala)
# 
# ### What we have done
# 
# 1. **Connected to SMHI Open Data API** (Meteorological Observations)
# 2. **Selected the best station for Uppsala**
#    - Station ID: **97510 - Uppsala Aut** (active automatic station)
#    - Rejected the old “Uppsala” station (inactive) and the airport station (less representative of the city)
# 
# 3. **Downloaded daily maximum temperature (parameter 20)**
#    - Period: **1 June 1985 - 30 June 2026** (~41 years)
#    - Total days: **15,004**
# 
# 4. **Cleaned the data**
#    - Columns kept: `date`, `tmax`, `quality`
# 
# 5. **Heatwave definitions aligned with Swedish practice**
#    - **Level 1 (Notice / health-relevant)**: Tmax ≥ **27 °C** for ≥ **3 consecutive days**
#    - **Level 2 (Warning / extreme)**: Tmax ≥ **30 °C** for ≥ **3 consecutive days**
#    - (Optional later: classic SMHI definition ≥ 25 °C for ≥ 5 days)
# 
# ### Why these thresholds?
# - Pure ≥ 30 °C is too rare in Uppsala and misses most periods that already increase health risk.
# - Swedish epidemiological studies show mortality rises clearly around 26-27 °C for a few consecutive days.
# - SMHI issues a “Meddelande om höga temperaturer” at ≈ 26 °C / 3 days and yellow warnings at 30 °C / 3-4 days.
# - This multi-level approach lets the future tool answer both everyday warm spells and rare extreme events.
# 
# ### Next steps
# - Run the Level 1 and Level 2 analysis side-by-side
# - Calculate frequency per decade
# - Build the first version of the deterministic risk function

# %%
def find_heatwaves(df, threshold, min_duration=3):
    """
    Find heatwave periods where Tmax >= threshold for at least min_duration consecutive days.
    Returns a DataFrame with start, end, duration, max_tmax, mean_tmax.
    """
    temp = df.copy()
    temp["is_hot"] = temp["tmax"] >= threshold

    # Group consecutive hot days
    temp["group"] = (temp["is_hot"] != temp["is_hot"].shift()).cumsum()
    temp.loc[~temp["is_hot"], "group"] = 0

    heatwaves = (
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

    # Keep only periods that meet the minimum duration
    heatwaves = heatwaves[heatwaves["duration"] >= min_duration].reset_index(drop=True)
    return heatwaves

# %%
# Level 1: ≥ 27 °C for ≥ 3 consecutive days (health-relevant / notice level)
hw_27 = find_heatwaves(df_hist_clean, threshold=27.0, min_duration=3)

# Level 2: ≥ 30 °C for ≥ 3 consecutive days (warning / extreme level)
hw_30 = find_heatwaves(df_hist_clean, threshold=30.0, min_duration=3)

print("=" * 60)
print("LEVEL 1 – Tmax ≥ 27 °C for ≥ 3 consecutive days")
print("=" * 60)
print(f"Number of heatwave periods: {len(hw_27)}")
print(hw_27.sort_values("start").to_string())
print("\nDuration statistics:")
print(hw_27["duration"].describe())

print("\n" + "=" * 60)
print("LEVEL 2 – Tmax ≥ 30 °C for ≥ 3 consecutive days")
print("=" * 60)
print(f"Number of heatwave periods: {len(hw_30)}")
print(hw_30.sort_values("start").to_string())
print("\nDuration statistics:")
print(hw_30["duration"].describe())

# %%
def add_decade(hw_df):
    hw = hw_df.copy()
    hw["decade"] = (hw["start"].dt.year // 10) * 10
    return hw.groupby("decade").size().rename("count")

print("Number of Level-1 heatwaves (≥27 °C, ≥3 days) per decade:")
print(add_decade(hw_27))

print("\nNumber of Level-2 heatwaves (≥30 °C, ≥3 days) per decade:")
print(add_decade(hw_30))

# %%
# Level 3: Classic SMHI definition – Tmax ≥ 25 °C for ≥ 5 consecutive days
hw_25 = find_heatwaves(df_hist_clean, threshold=25.0, min_duration=5)

print("=" * 60)
print("LEVEL 3 – Tmax ≥ 25 °C for ≥ 5 consecutive days (SMHI classic)")
print("=" * 60)
print(f"Number of heatwave periods: {len(hw_25)}")
print("\nDuration statistics:")
print(hw_25["duration"].describe())
print("\nNumber per decade:")
print(add_decade(hw_25))
print("\nLongest / hottest events:")
print(hw_25.sort_values("duration", ascending=False).head(8))

# %%
def compute_heatwave_risk(days: int, temp: float, df_hist=df_hist_clean):
    """
    Deterministic heatwave risk estimator for Uppsala (station 97510).

    Parameters
    ----------
    days : int
        Number of consecutive days expected
    temp : float
        Expected daily maximum temperature (°C)

    Returns
    -------
    dict with:
        - level: "none" / "level1" / "level2" / "level3"
        - description: short text
        - historical_count: how many similar events occurred 1985–2026
        - risk_score: simple empirical score (0–1)
        - comment: extra context
    """

    # Determine which level the query matches
    if temp >= 30 and days >= 3:
        level = "level2"
        description = f"Extreme heat (Level 2): ≥30°C for ≥3 days"
        # Count historical Level 2 events
        hist = find_heatwaves(df_hist, threshold=30.0, min_duration=3)
        historical_count = len(hist)
        # Simple risk score: rarer = higher relative risk
        risk_score = min(1.0, 0.4 + (historical_count / 20))
        comment = "This matches SMHI yellow-warning territory. Rare but high-impact in Uppsala."

    elif temp >= 27 and days >= 3:
        level = "level1"
        description = f"Health-relevant heat (Level 1): ≥27°C for ≥3 days"
        hist = find_heatwaves(df_hist, threshold=27.0, min_duration=3)
        historical_count = len(hist)
        risk_score = min(1.0, 0.25 + (historical_count / 80))
        comment = "Temperature range where Swedish studies show clear increases in mortality risk."

    elif temp >= 25 and days >= 5:
        level = "level3"
        description = f"Classic SMHI heatwave (Level 3): ≥25°C for ≥5 days"
        hist = find_heatwaves(df_hist, threshold=25.0, min_duration=5)
        historical_count = len(hist)
        risk_score = min(1.0, 0.15 + (historical_count / 120))
        comment = "Official SMHI climatological definition of a heatwave."

    else:
        level = "none"
        description = "Below defined heatwave thresholds"
        historical_count = 0
        risk_score = 0.05
        comment = "Does not meet Level 1, 2 or 3 criteria used in this tool."

    return {
        "level": level,
        "description": description,
        "historical_count": historical_count,
        "risk_score": round(risk_score, 2),
        "comment": comment,
        "query": f"{days} days at {temp}°C"
    }

# %%
# Test a few realistic queries
examples = [
    (3, 28),   # typical Level 1
    (4, 31),   # Level 2
    (6, 26),   # Level 3
    (2, 29),   # too short
    (5, 24),   # too cool
    (8, 32),   # strong Level 2 (like 2018)
]

print("Testing the risk function:\n")
for days, temp in examples:
    result = compute_heatwave_risk(days, temp)
    print(f"Query: {days} days at {temp}°C")
    print(f"  → Level: {result['level']}")
    print(f"  → {result['description']}")
    print(f"  → Historical similar events: {result['historical_count']}")
    print(f"  → Risk score: {result['risk_score']}")
    print(f"  → {result['comment']}")
    print("-" * 50)

# %% [markdown]
# ## Summary of this notebook
# 
# ### Goal
# Acquire and explore long-term daily maximum temperature data for Uppsala and define heatwave levels that are meaningful for local health risk.
# 
# ### Data
# - **Source**: SMHI Open Data (Meteorological Observations)
# - **Station**: 97510 - Uppsala Aut (active automatic station)
# - **Parameter**: Daily maximum temperature (ID 20)
# - **Period**: 1 June 1985 - 30 June 2026 (15 004 days)
# 
# ### Heatwave definitions used (aligned with Swedish practice)
# 
# | Level | Threshold              | Min. duration | Meaning |
# |-------|------------------------|---------------|---------|
# | **1** | ≥ 27 °C                | ≥ 3 days      | Health-relevant / notice level (mortality risk starts rising) |
# | **2** | ≥ 30 °C                | ≥ 3 days      | Extreme / SMHI yellow-warning level |
# | **3** | ≥ 25 °C                | ≥ 5 days      | Classic SMHI climatological heatwave |
# 
# ### Main findings
# - Level 1 events are relatively common and have increased over time.
# - Level 2 events are rare (only 13 in 41 years) but high-impact (especially 2018).
# - Level 3 captures longer warm periods, including the very long 2018 event (29 days).
# 
# ### Deliverable
# A first working version of the deterministic function `compute_heatwave_risk(days, temp)` that will be improved and used by the LLM agent in later notebooks.

# %%
# Save key objects so we can load them quickly in the next notebook
df_hist_clean.to_csv("uppsala_tmax_1985_2026.csv", index=False)

hw_27.to_csv("heatwaves_level1_27C.csv", index=False)
hw_30.to_csv("heatwaves_level2_30C.csv", index=False)
hw_25.to_csv("heatwaves_level3_25C.csv", index=False)

print("Files saved:")
print("  - uppsala_tmax_1985_2026.csv")
print("  - heatwaves_level1_27C.csv")
print("  - heatwaves_level2_30C.csv")
print("  - heatwaves_level3_25C.csv")


