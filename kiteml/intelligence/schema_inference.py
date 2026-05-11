"""
schema_inference.py — Full structural schema inference for KiteML.

Generates a rich internal schema for every column: nullability, cardinality,
value ranges, and distribution shape.  This schema drives preprocessing
decisions and reporting downstream.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class ColumnSchema:
    """Structural metadata for one column."""

    name: str
    dtype: str
    nullable: bool
    null_count: int
    null_ratio: float
    n_unique: int
    unique_ratio: float
    # Numeric only
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    skewness: Optional[float] = None
    distribution: Optional[str] = None  # "normal","right_skewed","left_skewed","uniform","bimodal"
    # Categorical only
    top_values: Optional[List[Any]] = None
    top_freqs: Optional[List[float]] = None


@dataclass
class DataSchema:
    """Full schema of a DataFrame."""

    n_rows: int
    n_cols: int
    columns: Dict[str, ColumnSchema]
    memory_bytes: int

    def to_dict(self) -> dict:
        out = {}
        for name, s in self.columns.items():
            out[name] = {k: v for k, v in s.__dict__.items() if v is not None}
        return out


def _distribution_label(series: pd.Series) -> str:
    """Heuristically label the distribution shape of a numeric series."""
    clean = series.dropna()
    if len(clean) < 10:
        return "insufficient_data"
    skew = float(clean.skew())
    kurt = float(clean.kurt())
    if abs(skew) < 0.5:
        if abs(kurt) < 1:
            return "normal"
        return "normal"
    if skew > 1.0:
        return "right_skewed"
    if skew < -1.0:
        return "left_skewed"
    # Bimodal: rough check with dip test approximation
    if kurt < -1.0:
        return "bimodal"
    return "slightly_skewed"


def _infer_column_schema(series: pd.Series, name: str) -> ColumnSchema:
    n_total = len(series)
    n_null = int(series.isna().sum())
    null_ratio = n_null / n_total if n_total > 0 else 0.0
    non_null = series.dropna()
    n_unique = int(non_null.nunique())
    unique_ratio = n_unique / len(non_null) if len(non_null) > 0 else 0.0

    schema = ColumnSchema(
        name=name,
        dtype=str(series.dtype),
        nullable=n_null > 0,
        null_count=n_null,
        null_ratio=round(null_ratio, 4),
        n_unique=n_unique,
        unique_ratio=round(unique_ratio, 4),
    )

    if pd.api.types.is_numeric_dtype(series):
        schema.min_val = float(non_null.min()) if len(non_null) > 0 else None
        schema.max_val = float(non_null.max()) if len(non_null) > 0 else None
        schema.mean = float(non_null.mean()) if len(non_null) > 0 else None
        schema.median = float(non_null.median()) if len(non_null) > 0 else None
        schema.std = float(non_null.std()) if len(non_null) > 1 else None
        schema.skewness = float(non_null.skew()) if len(non_null) > 3 else None
        schema.distribution = _distribution_label(series)
    else:
        vc = non_null.value_counts(normalize=True).head(5)
        schema.top_values = vc.index.tolist()
        schema.top_freqs = [round(float(f), 4) for f in vc.values.tolist()]

    return schema


def infer_schema(df: pd.DataFrame) -> DataSchema:
    """
    Infer the full structural schema of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    DataSchema
    """
    columns = {col: _infer_column_schema(df[col], col) for col in df.columns}
    return DataSchema(
        n_rows=len(df),
        n_cols=len(df.columns),
        columns=columns,
        memory_bytes=int(df.memory_usage(deep=True).sum()),
    )
