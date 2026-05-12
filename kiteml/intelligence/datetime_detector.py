"""
datetime_detector.py — Detect and extract datetime features from DataFrames.

Automatically:
  1. Detects datetime columns (by dtype or string parseability)
  2. Extracts temporal features: year, month, day, weekday, hour, quarter
  3. Detects seasonality patterns
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class DatetimeColumnInfo:
    """Metadata about a detected datetime column."""

    column: str
    original_dtype: str
    min_date: str | None
    max_date: str | None
    date_range_days: int | None
    extracted_features: list[str]  # which features will be extracted
    has_time_component: bool
    confidence: float


@dataclass
class DatetimeDetectionResult:
    """Result of datetime column detection."""

    datetime_columns: list[str]
    details: dict[str, DatetimeColumnInfo]
    has_datetime: bool


def _try_parse_datetime(series: pd.Series) -> pd.Series | None:
    """Return parsed datetime series or None if parsing fails."""
    try:
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().mean() > 0.8:
            return parsed
    except Exception:
        pass
    return None


def detect_datetime_columns(df: pd.DataFrame) -> DatetimeDetectionResult:
    """
    Detect datetime columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    DatetimeDetectionResult
    """
    dt_cols: list[str] = []
    details: dict[str, DatetimeColumnInfo] = {}

    for col in df.columns:
        series = df[col]
        parsed: pd.Series | None = None
        original_dtype = str(series.dtype)
        confidence = 0.0

        if pd.api.types.is_datetime64_any_dtype(series):
            parsed = series
            confidence = 0.99
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            parsed = _try_parse_datetime(series.dropna().head(200).astype(str))
            confidence = 0.85 if parsed is not None else 0.0

        if parsed is None or confidence < 0.5:
            continue

        non_null = parsed.dropna()
        min_date = str(non_null.min().date()) if len(non_null) > 0 else None
        max_date = str(non_null.max().date()) if len(non_null) > 0 else None
        date_range = int((non_null.max() - non_null.min()).days) if len(non_null) > 1 else None
        has_time = bool((non_null.dt.hour != 0).any()) if len(non_null) > 0 else False

        features = ["year", "month", "day", "dayofweek", "quarter"]
        if has_time:
            features += ["hour", "minute"]
        if date_range and date_range > 60:
            features.append("is_weekend")

        info = DatetimeColumnInfo(
            column=col,
            original_dtype=original_dtype,
            min_date=min_date,
            max_date=max_date,
            date_range_days=date_range,
            extracted_features=features,
            has_time_component=has_time,
            confidence=confidence,
        )
        dt_cols.append(col)
        details[col] = info

    return DatetimeDetectionResult(
        datetime_columns=dt_cols,
        details=details,
        has_datetime=len(dt_cols) > 0,
    )


def extract_datetime_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Extract temporal features from datetime columns and return augmented DataFrame.

    Original datetime columns are dropped; extracted features are added.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str
        Names of datetime columns to expand.

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime columns replaced by extracted numeric features.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        prefix = col
        df[f"{prefix}_year"] = parsed.dt.year
        df[f"{prefix}_month"] = parsed.dt.month
        df[f"{prefix}_day"] = parsed.dt.day
        df[f"{prefix}_dayofweek"] = parsed.dt.dayofweek
        df[f"{prefix}_quarter"] = parsed.dt.quarter
        if parsed.dt.hour.any():
            df[f"{prefix}_hour"] = parsed.dt.hour
        df.drop(columns=[col], inplace=True)
    return df
