"""
Datetime parsing helpers.

This project ingests timestamps in multiple formats depending on dataset/source:
- ISO8601 strings (often with a trailing 'Z')
- Unix timestamps in seconds / milliseconds / microseconds / nanoseconds
- Numeric timestamps stored as strings (object dtype)

Pandas' default `to_datetime` interpretation for integer-like strings is **nanoseconds**,
which can silently collapse real-world times into ~00:xx and break time-of-day features.
"""

from __future__ import annotations

import pandas as pd


def _infer_unix_unit_from_max(max_val: float) -> str:
    """
    Infer the unit for Unix timestamps by magnitude.

    Thresholds are intentionally broad:
    - ns: > 1e17  (e.g., 1700000000000000000)
    - us: > 1e14
    - ms: > 1e11
    - s : else
    """
    if max_val > 1e17:
        return "ns"
    if max_val > 1e14:
        return "us"
    if max_val > 1e11:
        return "ms"
    return "s"


def parse_timestamp_series(ts: pd.Series) -> pd.Series:
    """
    Parse a timestamp series into timezone-naive UTC `datetime64[ns]`.

    - Handles datetime-like, numeric, ISO8601 strings, and numeric strings.
    - Always parses/normalizes as UTC and then drops timezone info, so downstream
      operations like `.astype("int64")` and `.dt.hour` are reliable.
    """
    # Datetime dtype (naive or tz-aware)
    if pd.api.types.is_datetime64_any_dtype(ts):
        dt = pd.to_datetime(ts, errors="coerce", utc=True)
        return dt.dt.tz_convert(None)

    # Pure numeric dtype (including nullable Int64/Float64)
    if pd.api.types.is_numeric_dtype(ts):
        nums = pd.to_numeric(ts, errors="coerce")
        max_val = nums.max(skipna=True)
        if pd.isna(max_val):
            dt = pd.to_datetime(ts, errors="coerce", utc=True)
        else:
            unit = _infer_unix_unit_from_max(float(max_val))
            dt = pd.to_datetime(nums, unit=unit, errors="coerce", utc=True)
        return dt.dt.tz_convert(None)

    # Object/mixed dtype: try numeric first (covers "1704...618" stored as strings),
    # then fall back to ISO parsing for the remaining values.
    nums = pd.to_numeric(ts, errors="coerce")
    numeric_ratio = float(nums.notna().mean()) if len(nums) else 0.0

    if numeric_ratio > 0.0:
        max_val = nums.max(skipna=True)
        if pd.isna(max_val):
            dt_num = pd.to_datetime(nums, errors="coerce", utc=True)
        else:
            unit = _infer_unix_unit_from_max(float(max_val))
            dt_num = pd.to_datetime(nums, unit=unit, errors="coerce", utc=True)

        if numeric_ratio < 1.0:
            non_num = ts.where(nums.isna())
            dt_str = pd.to_datetime(non_num, errors="coerce", utc=True)
            dt = dt_num.fillna(dt_str)
        else:
            dt = dt_num
    else:
        dt = pd.to_datetime(ts, errors="coerce", utc=True)

    return dt.dt.tz_convert(None)

