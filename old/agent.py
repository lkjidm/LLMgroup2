
import pandas as pd
import numpy as np

# Load the historical data
df_hist = pd.read_csv("uppsala_tmax_1985_2026.csv", parse_dates=["date"])
print("Data loaded:", df_hist.shape)

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
    Deterministic heat-risk estimator for Uppsala with richer context.
    """
    # Input validation
    if not isinstance(days, (int, np.integer)) or days < 1:
        return {"level": "invalid", "description": "Invalid input", "historical_count": 0,
                "risk_score": 0.0, "comment": "Number of days must be an integer ≥ 1.",
                "extra_context": {}}
    if not isinstance(temp, (int, float, np.floating)) or temp < -40 or temp > 45:
        return {"level": "invalid", "description": "Invalid input", "historical_count": 0,
                "risk_score": 0.0, "comment": "Temperature must be a realistic value (°C).",
                "extra_context": {}}

    record_max = float(df_hist["tmax"].max())
    record_date = str(df_hist.loc[df_hist["tmax"].idxmax(), "date"].date())

    # Level 2 – Extreme multi-day
    if temp >= 30 and days >= 3:
        hist = find_heatwaves(df_hist, threshold=30.0, min_duration=3)
        count = len(hist)
        score = min(0.95, 0.60 + 0.025 * max(0, 15 - count))
        longest = hist.loc[hist["duration"].idxmax()] if len(hist) > 0 else None
        return {
            "level": "level2",
            "description": "Extreme heat (Level 2): ≥30°C for ≥3 days",
            "historical_count": count,
            "risk_score": float(round(score, 2)),
            "comment": "Matches SMHI yellow-warning level. Rare but high health impact in Uppsala.",
            "extra_context": {
                "record_max_tmax": record_max,
                "record_date": record_date,
                "longest_similar_event": {
                    "start": str(longest["start"].date()),
                    "end": str(longest["end"].date()),
                    "duration": int(longest["duration"]),
                    "max_tmax": float(longest["max_tmax"])
                } if longest is not None else None
            }
        }

    # Extreme single day
    if temp >= 33 and days >= 1:
        extreme_days = int((df_hist["tmax"] >= 33).sum())
        score = min(0.75, 0.45 + 0.02 * extreme_days)
        return {
            "level": "extreme_single",
            "description": "Extreme single day: ≥33°C (even for 1 day)",
            "historical_count": extreme_days,
            "risk_score": float(round(score, 2)),
            "comment": "Very high temperature for Uppsala. Notable even as a single day.",
            "extra_context": {
                "record_max_tmax": record_max,
                "record_date": record_date
            }
        }

    # Level 1 – Health-relevant
    if temp >= 27 and days >= 3:
        hist = find_heatwaves(df_hist, threshold=27.0, min_duration=3)
        count = len(hist)
        score = min(0.80, 0.30 + 0.006 * count)
        return {
            "level": "level1",
            "description": "Health-relevant heat (Level 1): ≥27°C for ≥3 days",
            "historical_count": count,
            "risk_score": float(round(score, 2)),
            "comment": "Temperature range where Swedish studies show clear increases in mortality risk.",
            "extra_context": {
                "record_max_tmax": record_max,
                "record_date": record_date
            }
        }

    # Level 3 – Classic SMHI
    if temp >= 25 and days >= 5:
        hist = find_heatwaves(df_hist, threshold=25.0, min_duration=5)
        count = len(hist)
        score = min(0.65, 0.18 + 0.004 * count)
        return {
            "level": "level3",
            "description": "Classic SMHI heatwave (Level 3): ≥25°C for ≥5 days",
            "historical_count": count,
            "risk_score": float(round(score, 2)),
            "comment": "Official SMHI climatological definition of a heatwave.",
            "extra_context": {
                "record_max_tmax": record_max,
                "record_date": record_date
            }
        }

    # Below thresholds
    return {
        "level": "none",
        "description": "Below defined heatwave thresholds",
        "historical_count": 0,
        "risk_score": 0.05,
        "comment": "Does not meet any of the defined heat criteria.",
        "extra_context": {
            "record_max_tmax": record_max,
            "record_date": record_date
        }
    }

# %%
def heatwave_risk_tool(days: int, temperature: float) -> dict:
    """
    Estimate heatwave risk for Uppsala given a number of consecutive days
    and an expected daily maximum temperature.

    Uses historical observations from Uppsala Aut (1985–2026) and
    Swedish heat definitions (SMHI + epidemiological thresholds).

    Args:
        days: Number of consecutive days (integer ≥ 1)
        temperature: Expected daily maximum temperature in °C

    Returns:
        Dictionary with level, description, historical_count, risk_score,
        comment, and extra_context (record temperature, etc.)
    """
    return compute_heatwave_risk(days=days, temp=temperature, df_hist=df_hist)

# %%
print(heatwave_risk_tool(4, 31))
print(heatwave_risk_tool(1, 34))


# %%
# Tell Gemini about our tool
heatwave_tool_declaration = types.FunctionDeclaration(
    name="heatwave_risk_tool",
    description="Estimate heatwave risk for Uppsala given number of consecutive days and expected maximum temperature. Uses historical data 1985-2026 and Swedish heat definitions.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "days": types.Schema(
                type=types.Type.INTEGER,
                description="Number of consecutive days (integer ≥ 1)"
            ),
            "temperature": types.Schema(
                type=types.Type.NUMBER,
                description="Expected daily maximum temperature in degrees Celsius"
            ),
        },
        required=["days", "temperature"]
    )
)

tools = types.Tool(function_declarations=[heatwave_tool_declaration])

# %%
SYSTEM_PROMPT = """
You are a careful climate-health assistant focused on Uppsala County, Sweden.
Your name is CLIMES-Monitor.

Rules you must follow:
1. When a question involves temperature, number of days, or heat risk, you MUST call the heatwave_risk_tool.
2. Never invent numbers. Only use the values returned by the tool.
3. Always mention the risk level and the historical count from the tool.
4. Keep answers clear, factual and relevant for public-health decision making.
5. If the tool returns extra_context (record temperature, longest similar event), you may use it.
6. Do not discuss topics unrelated to climate, heat or health impacts in Uppsala.
"""

def ask_heatwave_agent(user_question: str):
    """
    More robust agent with system prompt and richer tool results.
    """
    # First call
    response = client.models.generate_content(
        model="gemini-3.6-flash",          # or the model you are using (e.g. gemini-2.5-flash / gemini-3.6-flash)
        contents=user_question,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[tools],
            temperature=0.15
        )
    )

    candidate = response.candidates[0]

    # Did the model request the tool?
    if candidate.content.parts and candidate.content.parts[0].function_call:
        fc = candidate.content.parts[0].function_call
        print(f"→ Model called tool: {fc.name}")
        print(f"→ Arguments: {dict(fc.args)}")

        # Run the real tool
        tool_result = heatwave_risk_tool(
            days=int(fc.args.get("days", 1)),
            temperature=float(fc.args.get("temperature", 0))
        )
        print("→ Tool result:", tool_result)

        # Second call – give the tool result back
        response2 = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[
                user_question,
                candidate.content,
                types.Part.from_function_response(
                    name="heatwave_risk_tool",
                    response=tool_result
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.25
            )
        )
        return response2.text

    # Model answered without the tool
    return response.text

# %%
print(ask_heatwave_agent("What is the heat risk if Uppsala has 4 days with maximum temperatures around 31 degrees?"))
print("\n" + "="*70 + "\n")
print(ask_heatwave_agent("How serious is a single day at 34°C in Uppsala?"))
print("\n" + "="*70 + "\n")
print(ask_heatwave_agent("Is 3 days at 28°C something we should worry about in Uppsala?"))


