"""Descriptive counts of consecutive hot-day runs in the bundled SMHI series.

Counts use the user's exact threshold/duration. They are historical observations,
not official warning definitions, forecasts, or probabilities of future events.
"""
from pathlib import Path
from numbers import Real, Integral
import math
import pandas as pd

_CSV = Path(__file__).resolve().parent / 'uppsala_tmax_1985_2026.csv'
_df = None


def _prepare(frame):
    if frame.empty or not {'date', 'tmax'}.issubset(frame.columns):
        raise ValueError('Missing historical dates or daily maximum temperatures')
    data = frame.copy()
    data['date'] = pd.to_datetime(data['date'], errors='coerce').dt.normalize()
    if data['date'].isna().any() or data['date'].duplicated().any():
        raise ValueError('Invalid or duplicate historical dates')
    data['tmax'] = pd.to_numeric(data['tmax'], errors='coerce')
    data.loc[~data['tmax'].map(lambda x: math.isfinite(x)), 'tmax'] = float('nan')
    if not data['tmax'].notna().any():
        raise ValueError('No valid historical temperatures')
    return data.sort_values('date').reset_index(drop=True)


def _load():
    global _df
    if _df is None:
        _df = _prepare(pd.read_csv(_CSV))
    return _df


def find_heatwaves(df, threshold, min_duration=3):
    if isinstance(threshold, bool) or not isinstance(threshold, Real) or not math.isfinite(threshold):
        raise ValueError('Threshold must be finite')
    if isinstance(min_duration, bool) or not isinstance(min_duration, Integral) or min_duration < 1:
        raise ValueError('Duration must be a positive integer')
    data = _prepare(df)
    hot = data['tmax'].ge(threshold)
    consecutive = data['date'].diff().dt.days.eq(1)
    # A missing date or missing temperature breaks a consecutive-day run.
    data['group'] = (hot.ne(hot.shift()) | ~consecutive).cumsum()
    events = data[hot].groupby('group').agg(
        start=('date', 'min'), end=('date', 'max'),
        duration=('date', 'count'), max_tmax=('tmax', 'max'),
    ).reset_index(drop=True)
    return events[events['duration'].ge(min_duration)].reset_index(drop=True)


def historical_heatwave_stats(days: int, temp: float) -> dict:
    if isinstance(days, bool) or not isinstance(days, Integral) or days < 1:
        return {'error': 'days must be an integer >= 1'}
    if isinstance(temp, bool) or not isinstance(temp, Real) or not math.isfinite(temp) or not -40 <= temp <= 45:
        return {'error': 'temperature must be a finite Celsius value from -40 to 45'}
    try:
        data = _load()
        events = find_heatwaves(data, float(temp), int(days))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {'error': 'Historical observations unavailable', 'reason': type(exc).__name__}
    first, last = data['date'].min(), data['date'].max()
    calendar_days = (last - first).days + 1
    years = calendar_days / 365.25
    longest = None
    if not events.empty:
        row = events.loc[events['duration'].idxmax()]
        longest = {'start': str(row['start'].date()), 'end': str(row['end'].date()),
                   'duration': int(row['duration']), 'max_tmax': float(row['max_tmax'])}
    return {
        'label': f'Daily maximum >= {temp:g} C for at least {days} consecutive days',
        'query_days': int(days), 'query_temp': float(temp),
        'threshold_used': float(temp), 'minimum_duration_used': int(days),
        'historical_count': len(events), 'years_covered': round(years, 3),
        'empirical_rate_per_year': round(len(events) / years, 4),
        'counting_method': 'Maximal non-overlapping consecutive-day runs; gaps break runs',
        'longest_similar': longest, 'record_max_tmax': float(data['tmax'].max()),
        'valid_observation_days': int(data['tmax'].notna().sum()),
        'missing_calendar_days': calendar_days - len(data),
        'missing_temperature_days': int(data['tmax'].isna().sum()),
        'station': 'Uppsala Aut (97510)',
        'period': f'{first.date()} to {last.date()}',
    }
