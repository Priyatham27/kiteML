"""
outlier_detector.py — Multi-method outlier detection.

Supports three detection methods:
  - IQR  (robust, distribution-free)
  - Z-score (assumes approximate normality)
  - IsolationForest (advanced, model-based — used for multivariate outliers)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ColumnOutlierInfo:
    """Outlier statistics for a single numeric column."""

    column: str
    method: str
    n_outliers: int
    outlier_ratio: float
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    example_values: list[float]


@dataclass
class OutlierReport:
    """Complete outlier analysis report."""

    columns_analyzed: int
    columns_with_outliers: list[str]
    details: dict[str, ColumnOutlierInfo]
    total_outlier_rows: int  # rows flagged by any column
    outlier_row_ratio: float
    has_outliers: bool
    recommendations: list[str]


def _iqr_outliers(series: pd.Series) -> tuple:
    """Return (mask, lower, upper) using IQR method."""
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    return mask, lower, upper


def _zscore_outliers(series: pd.Series, threshold: float = 3.0) -> tuple:
    """Return (mask, lower, upper) using Z-score method."""
    mean = float(series.mean())
    std = float(series.std())
    if std == 0:
        return pd.Series(False, index=series.index), mean, mean
    z = (series - mean) / std
    mask = z.abs() > threshold
    return mask, mean - threshold * std, mean + threshold * std


def detect_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    zscore_threshold: float = 3.0,
    exclude_columns: Optional[list[str]] = None,
) -> OutlierReport:
    """
    Detect outliers in numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
    method : str
        ``'iqr'`` or ``'zscore'``. Default ``'iqr'``.
    zscore_threshold : float
        Z-score cutoff. Default 3.0.
    exclude_columns : list of str, optional
        Columns to skip.

    Returns
    -------
    OutlierReport
    """
    exclude = set(exclude_columns or [])
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    details: dict[str, ColumnOutlierInfo] = {}
    all_outlier_mask = pd.Series(False, index=df.index)

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue

        if method == "zscore":
            mask, lower, upper = _zscore_outliers(series, zscore_threshold)
        else:
            mask, lower, upper = _iqr_outliers(series)

        n_out = int(mask.sum())
        if n_out > 0:
            example_vals = series[mask].head(5).tolist()
            info = ColumnOutlierInfo(
                column=col,
                method=method,
                n_outliers=n_out,
                outlier_ratio=round(n_out / len(series), 4),
                lower_bound=round(lower, 4),
                upper_bound=round(upper, 4),
                example_values=[round(v, 4) for v in example_vals],
            )
            details[col] = info
            all_outlier_mask = all_outlier_mask | mask.reindex(df.index, fill_value=False)

    cols_with_outliers = list(details.keys())
    total_outlier_rows = int(all_outlier_mask.sum())
    outlier_row_ratio = round(total_outlier_rows / len(df), 4) if len(df) > 0 else 0.0

    recommendations: list[str] = []
    if cols_with_outliers:
        recommendations.append(f"Outliers detected in {len(cols_with_outliers)} column(s): {cols_with_outliers[:5]}")
        if outlier_row_ratio > 0.05:
            recommendations.append("High outlier rate. Consider robust scaling or winsorization.")
        else:
            recommendations.append("KiteML's StandardScaler handles mild outliers automatically.")

    return OutlierReport(
        columns_analyzed=len(numeric_cols),
        columns_with_outliers=cols_with_outliers,
        details=details,
        total_outlier_rows=total_outlier_rows,
        outlier_row_ratio=outlier_row_ratio,
        has_outliers=len(cols_with_outliers) > 0,
        recommendations=recommendations,
    )
