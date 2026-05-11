"""
memory_optimizer.py — Memory usage analysis and dtype optimization suggestions.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class ColumnMemoryInfo:
    column: str
    current_dtype: str
    current_bytes: int
    suggested_dtype: str
    estimated_savings_bytes: int
    reason: str


@dataclass
class MemoryReport:
    total_memory_bytes: int
    total_memory_mb: float
    potential_savings_bytes: int
    potential_savings_mb: float
    columns: Dict[str, ColumnMemoryInfo]
    recommendations: List[str]


def analyze_memory(df: pd.DataFrame) -> MemoryReport:
    """
    Analyze DataFrame memory usage and suggest dtype optimizations.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    MemoryReport
    """
    total_bytes = int(df.memory_usage(deep=True).sum())
    columns: Dict[str, ColumnMemoryInfo] = {}
    total_savings = 0
    recommendations: List[str] = []

    for col in df.columns:
        series = df[col]
        current_dtype = str(series.dtype)
        col_bytes = int(series.memory_usage(deep=True))
        suggested_dtype = current_dtype
        savings = 0
        reason = "no optimization available"

        if pd.api.types.is_integer_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                col_min, col_max = int(non_null.min()), int(non_null.max())
                for dtype in [np.int8, np.int16, np.int32]:
                    info = np.iinfo(dtype)
                    if info.min <= col_min and col_max <= info.max:
                        new_bytes = int(series.astype(dtype).memory_usage(deep=True))
                        if new_bytes < col_bytes:
                            savings = col_bytes - new_bytes
                            suggested_dtype = dtype.__name__
                            reason = f"range [{col_min}, {col_max}] fits in {dtype.__name__}"
                            break

        elif pd.api.types.is_float_dtype(series):
            if current_dtype == "float64":
                new_bytes = int(series.astype(np.float32).memory_usage(deep=True))
                if new_bytes < col_bytes:
                    savings = col_bytes - new_bytes
                    suggested_dtype = "float32"
                    reason = "downcast float64 → float32 (minor precision loss)"

        elif pd.api.types.is_object_dtype(series):
            n_unique = series.nunique()
            if n_unique / len(series) < 0.5:
                new_bytes = int(series.astype("category").memory_usage(deep=True))
                if new_bytes < col_bytes:
                    savings = col_bytes - new_bytes
                    suggested_dtype = "category"
                    reason = f"low cardinality ({n_unique} unique) → category dtype"

        columns[col] = ColumnMemoryInfo(
            column=col,
            current_dtype=current_dtype,
            current_bytes=col_bytes,
            suggested_dtype=suggested_dtype,
            estimated_savings_bytes=savings,
            reason=reason,
        )
        total_savings += savings

    if total_savings > 0:
        savings_mb = total_savings / 1e6
        recommendations.append(f"Potential memory saving: {savings_mb:.1f} MB via dtype optimization.")
        top_cols = sorted(columns.values(), key=lambda c: c.estimated_savings_bytes, reverse=True)[:3]
        for ci in top_cols:
            if ci.estimated_savings_bytes > 0:
                recommendations.append(
                    f"  '{ci.column}': {ci.current_dtype} → {ci.suggested_dtype} "
                    f"(saves {ci.estimated_savings_bytes/1024:.0f} KB)"
                )

    return MemoryReport(
        total_memory_bytes=total_bytes,
        total_memory_mb=round(total_bytes / 1e6, 2),
        potential_savings_bytes=total_savings,
        potential_savings_mb=round(total_savings / 1e6, 2),
        columns=columns,
        recommendations=recommendations,
    )
