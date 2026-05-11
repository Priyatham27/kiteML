"""
column_analyzer.py — Deep column type classification for KiteML.

Detects the semantic type of every column in a DataFrame, going far beyond
dtype alone.  A column of dtype int64 could be a numerical measurement, an
ordinal category, an identifier, or a boolean — this module tells the
difference.

Column Types
------------
numerical   → continuous numbers (age, salary, temperature)
categorical → low-cardinality discrete values (gender, country)
ordinal     → ordered discrete values (low/medium/high)
boolean     → binary true/false or 0/1
datetime    → timestamps / date strings
text        → free-form natural language
identifier  → unique row IDs (customer_id, order_id)
constant    → single-value useless columns
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd


class ColumnType(str, Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    IDENTIFIER = "identifier"
    CONSTANT = "constant"
    UNKNOWN = "unknown"


@dataclass
class ColumnProfile:
    """Semantic profile of a single DataFrame column."""

    name: str
    dtype: str
    column_type: ColumnType
    n_unique: int
    unique_ratio: float
    null_ratio: float
    sample_values: List
    confidence: float  # 0–1 confidence in the inferred type
    notes: List[str] = field(default_factory=list)


@dataclass
class ColumnAnalysisResult:
    """Full analysis of all columns in a DataFrame."""

    profiles: Dict[str, ColumnProfile]
    type_summary: Dict[str, int]  # ColumnType → count

    def of_type(self, column_type: ColumnType) -> List[str]:
        """Return column names matching the given type."""
        return [name for name, p in self.profiles.items() if p.column_type == column_type]

    def to_dict(self) -> dict:
        return {
            name: {
                "type": p.column_type.value,
                "dtype": p.dtype,
                "n_unique": p.n_unique,
                "unique_ratio": round(p.unique_ratio, 4),
                "null_ratio": round(p.null_ratio, 4),
                "confidence": round(p.confidence, 2),
                "notes": p.notes,
            }
            for name, p in self.profiles.items()
        }


# ── Ordinal hint vocabulary ──────────────────────────────────────────────────
_ORDINAL_SETS = [
    {"low", "medium", "high"},
    {"small", "medium", "large"},
    {"poor", "fair", "good", "excellent"},
    {"never", "rarely", "sometimes", "often", "always"},
    {"strongly disagree", "disagree", "neutral", "agree", "strongly agree"},
    {"very low", "low", "moderate", "high", "very high"},
    {"bronze", "silver", "gold", "platinum"},
]

# ── Identifier name hints ────────────────────────────────────────────────────
_ID_KEYWORDS = {"id", "uuid", "guid", "key", "index", "code", "ref", "no", "num", "number", "sku"}


def _is_datetime_parseable(series: pd.Series, sample_size: int = 50) -> bool:
    """Try to parse a sample of string values as datetimes."""
    sample = series.dropna().head(sample_size).astype(str)
    if len(sample) == 0:
        return False
    try:
        pd.to_datetime(sample, errors="raise")
        return True
    except Exception:
        parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().mean() > 0.8


def _avg_word_count(series: pd.Series, sample_size: int = 100) -> float:
    """Mean word count in a string column sample."""
    sample = series.dropna().head(sample_size).astype(str)
    if len(sample) == 0:
        return 0.0
    return float(sample.str.split().str.len().mean())


def _classify_column(series: pd.Series, name: str) -> ColumnProfile:
    """Infer the semantic type of a single column."""
    n_total = len(series)
    n_null = int(series.isna().sum())
    null_ratio = n_null / n_total if n_total > 0 else 0.0
    non_null = series.dropna()
    n_unique = int(non_null.nunique())
    unique_ratio = n_unique / len(non_null) if len(non_null) > 0 else 0.0
    sample_vals = non_null.head(5).tolist()
    dtype_str = str(series.dtype)
    notes: List[str] = []
    confidence = 0.9

    # ── CONSTANT ─────────────────────────────────────────────────────────
    if n_unique <= 1:
        return ColumnProfile(
            name,
            dtype_str,
            ColumnType.CONSTANT,
            n_unique,
            unique_ratio,
            null_ratio,
            sample_vals,
            confidence=0.99,
            notes=["single unique value"],
        )

    # ── DATETIME (dtype) ──────────────────────────────────────────────────
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnProfile(
            name,
            dtype_str,
            ColumnType.DATETIME,
            n_unique,
            unique_ratio,
            null_ratio,
            sample_vals,
            confidence=0.99,
            notes=["datetime64 dtype"],
        )

    # ── BOOLEAN ──────────────────────────────────────────────────────────
    if pd.api.types.is_bool_dtype(series):
        return ColumnProfile(
            name,
            dtype_str,
            ColumnType.BOOLEAN,
            n_unique,
            unique_ratio,
            null_ratio,
            sample_vals,
            confidence=0.99,
            notes=["bool dtype"],
        )

    if n_unique == 2:
        vals = set(non_null.unique())
        if vals <= {0, 1} or vals <= {True, False} or vals <= {"yes", "no"} or vals <= {"y", "n"}:
            notes.append("binary values")
            return ColumnProfile(
                name,
                dtype_str,
                ColumnType.BOOLEAN,
                n_unique,
                unique_ratio,
                null_ratio,
                sample_vals,
                confidence=0.95,
                notes=notes,
            )

    # ── STRING / OBJECT branch ────────────────────────────────────────────
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        str_series = non_null.astype(str)

        # Datetime string detection
        if _is_datetime_parseable(str_series):
            notes.append("parseable as datetime")
            return ColumnProfile(
                name,
                dtype_str,
                ColumnType.DATETIME,
                n_unique,
                unique_ratio,
                null_ratio,
                sample_vals,
                confidence=0.85,
                notes=notes,
            )

        # Text / NLP detection
        avg_words = _avg_word_count(str_series)
        if avg_words > 5 or (unique_ratio > 0.9 and avg_words > 2):
            notes.append(f"avg {avg_words:.1f} words per value")
            return ColumnProfile(
                name,
                dtype_str,
                ColumnType.TEXT,
                n_unique,
                unique_ratio,
                null_ratio,
                sample_vals,
                confidence=0.80,
                notes=notes,
            )

        # Ordinal detection
        lower_vals = {str(v).strip().lower() for v in non_null.unique()}
        for ordinal_set in _ORDINAL_SETS:
            if lower_vals <= ordinal_set or ordinal_set <= lower_vals:
                notes.append("matches ordinal vocabulary")
                return ColumnProfile(
                    name,
                    dtype_str,
                    ColumnType.ORDINAL,
                    n_unique,
                    unique_ratio,
                    null_ratio,
                    sample_vals,
                    confidence=0.88,
                    notes=notes,
                )

        # Identifier: name hints + near-unique
        name_lower = name.lower().replace(" ", "_")
        if unique_ratio > 0.95 and any(kw in name_lower for kw in _ID_KEYWORDS):
            notes.append("high uniqueness + ID name pattern")
            return ColumnProfile(
                name,
                dtype_str,
                ColumnType.IDENTIFIER,
                n_unique,
                unique_ratio,
                null_ratio,
                sample_vals,
                confidence=0.90,
                notes=notes,
            )

        # Categorical
        notes.append(f"{n_unique} unique values")
        return ColumnProfile(
            name,
            dtype_str,
            ColumnType.CATEGORICAL,
            n_unique,
            unique_ratio,
            null_ratio,
            sample_vals,
            confidence=0.85,
            notes=notes,
        )

    # ── NUMERIC branch ────────────────────────────────────────────────────
    if pd.api.types.is_numeric_dtype(series):
        # Identifier: integer + near-unique + name hint
        name_lower = name.lower().replace(" ", "_")
        if (
            pd.api.types.is_integer_dtype(series)
            and unique_ratio > 0.95
            and any(kw in name_lower for kw in _ID_KEYWORDS)
        ):
            notes.append("integer identifier pattern")
            return ColumnProfile(
                name,
                dtype_str,
                ColumnType.IDENTIFIER,
                n_unique,
                unique_ratio,
                null_ratio,
                sample_vals,
                confidence=0.88,
                notes=notes,
            )

        # Categorical numeric: low cardinality integers
        if pd.api.types.is_integer_dtype(series) and n_unique <= 20 and unique_ratio < 0.05:
            notes.append(f"low-cardinality integer ({n_unique} values)")
            return ColumnProfile(
                name,
                dtype_str,
                ColumnType.CATEGORICAL,
                n_unique,
                unique_ratio,
                null_ratio,
                sample_vals,
                confidence=0.75,
                notes=notes,
            )

        notes.append("continuous numeric")
        return ColumnProfile(
            name,
            dtype_str,
            ColumnType.NUMERICAL,
            n_unique,
            unique_ratio,
            null_ratio,
            sample_vals,
            confidence=0.92,
            notes=notes,
        )

    return ColumnProfile(
        name,
        dtype_str,
        ColumnType.UNKNOWN,
        n_unique,
        unique_ratio,
        null_ratio,
        sample_vals,
        confidence=0.0,
        notes=["unrecognized dtype"],
    )


def analyze_columns(
    df: pd.DataFrame,
    exclude: Optional[List[str]] = None,
) -> ColumnAnalysisResult:
    """
    Analyze every column in a DataFrame and return semantic type profiles.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze (target column may be included or excluded).
    exclude : list of str, optional
        Column names to skip (e.g. the target column).

    Returns
    -------
    ColumnAnalysisResult
        Structured result with per-column profiles and a type summary.
    """
    exclude_set = set(exclude or [])
    profiles: Dict[str, ColumnProfile] = {}

    for col in df.columns:
        if col in exclude_set:
            continue
        profiles[col] = _classify_column(df[col], col)

    type_summary: Dict[str, int] = {}
    for p in profiles.values():
        key = p.column_type.value
        type_summary[key] = type_summary.get(key, 0) + 1

    return ColumnAnalysisResult(profiles=profiles, type_summary=type_summary)
