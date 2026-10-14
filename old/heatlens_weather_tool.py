"""HeatLens integration tool: exact-threshold historical comparison.

Uses a team-provided CSV, not a live forecast. Missing observations break
confirmed runs. Counts refer to maximal qualifying observed runs; missing data
can split a true event. Quality flags are retained but not interpreted here.
The data owner must review cleaning, quality handling and source metadata.
"""
import csv
import math
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

_DATA = None
_METADATA = None


def load_weather_data(csv_path):
    """Load one observation per date; reject duplicates for owner review."""
    global _DATA, _METADATA
    observations = {}
    quality = Counter()
    invalid_temperatures = 0
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"date", "tmax"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV must contain date and tmax columns.")
        for row in reader:
            day = datetime.fromisoformat(row["date"].strip()).date()
            if day in observations:
                raise ValueError(f"Duplicate date {day}; ask the data owner to resolve it.")
            try:
                temperature = float(row["tmax"])
            except (TypeError, ValueError):
                temperature = None
            if temperature is None or not math.isfinite(temperature):
                temperature = None
                invalid_temperatures += 1
            observations[day] = temperature
            quality[row.get("quality", "unspecified")] += 1
    if not observations:
        raise ValueError("The CSV has no observations.")
    data = sorted(observations.items())
    valid = [(day, temp) for day, temp in data if temp is not None]
    if not valid:
        raise ValueError("The CSV has no finite temperatures.")
    start, end = data[0][0], data[-1][0]
    calendar_days = (end - start).days + 1
    maximum_day, maximum = max(valid, key=lambda item: item[1])
    metadata = {
        "station": "Uppsala Aut (97510), as identified by the team dataset",
        "variable": "Daily maximum temperature in degrees Celsius",
        "dataset_start": start.isoformat(),
        "dataset_end": end.isoformat(),
        "observation_rows": len(data),
        "missing_dates": calendar_days - len(data),
        "invalid_temperature_rows": invalid_temperatures,
        "quality_flag_counts": dict(quality),
        "quality_policy": "Retained as supplied; flag interpretation and filtering require data-owner review.",
        "record_max_tmax": maximum,
        "record_date": maximum_day.isoformat(),
    }
    _DATA, _METADATA = data, metadata
    return dict(metadata)


def heatwave_risk_tool(days, temperature):
    """Compare the exact requested threshold/duration with observed runs.

    The function name preserves the team's integration interface. Its output
    is a historical comparison, not a health-risk score or official warning.
    """
    if type(days) is not int or days < 1:
        raise ValueError("days must be a positive integer.")
    if type(temperature) not in (int, float) or not math.isfinite(temperature):
        raise ValueError("temperature must be a finite Celsius number.")
    if _DATA is None:
        raise RuntimeError("Call load_weather_data(csv_path) first.")

    events = []
    run = []

    def finish_run():
        if len(run) >= days:
            events.append({
                "start": run[0][0].isoformat(),
                "end": run[-1][0].isoformat(),
                "duration": len(run),
                "max_tmax": max(temp for _, temp in run),
            })
        run.clear()

    for day, temp in _DATA:
        if temp is None or temp < temperature:
            finish_run()
            continue
        if run and day != run[-1][0] + timedelta(days=1):
            finish_run()
        run.append((day, temp))
    finish_run()

    return {
        "comparison_type": "Historical temperature scenario comparison",
        "days": days,
        "temperature": temperature,
        "historical_count": len(events),
        "count_unit": "Maximal qualifying runs in the available consecutive daily observations",
        "historical_threshold_celsius": temperature,
        "historical_min_duration_days": days,
        "station": _METADATA["station"],
        "dataset_start": _METADATA["dataset_start"],
        "dataset_end": _METADATA["dataset_end"],
        "record_max_tmax": _METADATA["record_max_tmax"],
        "record_date": _METADATA["record_date"],
        "longest_matching_event": max(events, key=lambda event: event["duration"]) if events else None,
        "limitations": [
            f"Missing dates: {_METADATA['missing_dates']}; invalid temperature rows: {_METADATA['invalid_temperature_rows']}.",
            "Missing observations interrupt confirmed runs and may split a true event.",
            "Quality flags are retained pending data-owner review.",
            "The station dataset is not a forecast, calibrated health probability, or official warning.",
        ],
    }
