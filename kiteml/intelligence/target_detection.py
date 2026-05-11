"""
target_detection.py — Intelligent target column detection.

Infers the most likely target column without requiring the user to specify it
explicitly.  Uses a multi-signal scoring system rather than a single heuristic.

Signals (weighted)
------------------
1. Column position (last column → strong prior)
2. Keyword matching  (target, label, price, survived, churn, etc.)
3. Uniqueness ratio  (very high → likely continuous target or ID)
4. Correlation structure  (correlated with many features → likely target)
5. dtype semantics  (int/float vs object)
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

_TARGET_KEYWORDS = {
    # General
    "target",
    "label",
    "class",
    "output",
    "result",
    "response",
    "outcome",
    "dependent",
    "y",
    # Domain-specific
    "price",
    "cost",
    "salary",
    "revenue",
    "sales",
    "profit",
    "survived",
    "survival",
    "churn",
    "default",
    "fraud",
    "rating",
    "score",
    "grade",
    "rank",
    "category",
    "species",
    "diagnosis",
    "disease",
    "status",
    "approved",
    "converted",
}


@dataclass
class TargetDetectionResult:
    """Result of automatic target column detection."""

    column: str
    confidence: float  # 0–1
    reason: List[str]  # human-readable explanation of signals
    is_ambiguous: bool = False  # True when multiple columns score closely
    alternatives: List[str] = field(default_factory=list)


def _score_column(
    series: pd.Series,
    name: str,
    df: pd.DataFrame,
    is_last: bool,
) -> float:
    """Return a 0–1 likelihood score that this column is the target."""
    score = 0.0

    # Signal 1 — position (last column is conventional)
    if is_last:
        score += 0.35

    # Signal 2 — keyword match
    name_lower = name.lower().replace(" ", "_").replace("-", "_")
    if any(
        kw == name_lower or name_lower.endswith("_" + kw) or name_lower.startswith(kw + "_") for kw in _TARGET_KEYWORDS
    ):
        score += 0.30
    elif any(kw in name_lower for kw in _TARGET_KEYWORDS):
        score += 0.15

    # Signal 3 — cardinality (targets usually have fewer unique values relative to rows)
    n_total = len(series)
    non_null = series.dropna()
    n_unique = non_null.nunique()
    unique_ratio = n_unique / n_total if n_total > 0 else 1.0

    if unique_ratio < 0.10:
        score += 0.15  # discrete — good target candidate
    elif unique_ratio < 0.50:
        score += 0.05

    # Signal 4 — dtype: numeric or categorical both OK; pure text or IDs not
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_object_dtype(series):
        score += 0.05

    # Signal 5 — mean absolute correlation with other numeric columns
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if name in numeric_cols and len(numeric_cols) > 1:
            others = [c for c in numeric_cols if c != name]
            corrs = df[others].corrwith(series).abs().dropna()
            if len(corrs) > 0:
                mean_corr = float(corrs.mean())
                # Moderate correlation is a good target signal
                if 0.1 < mean_corr < 0.9:
                    score += 0.10
    except Exception:
        pass

    return min(score, 1.0)


def detect_target(df: pd.DataFrame) -> TargetDetectionResult:
    """
    Infer the most likely target column from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (all columns, no pre-filtering).

    Returns
    -------
    TargetDetectionResult
    """
    if len(df.columns) == 0:
        raise ValueError("DataFrame has no columns.")

    columns = list(df.columns)
    scores = {}

    for i, col in enumerate(columns):
        is_last = i == len(columns) - 1
        scores[col] = _score_column(df[col], col, df, is_last)

    sorted_cols = sorted(scores, key=lambda c: scores[c], reverse=True)
    best_col = sorted_cols[0]
    best_score = scores[best_col]

    # Build reason list
    reasons = []
    name_lower = best_col.lower().replace(" ", "_").replace("-", "_")
    if best_col == columns[-1]:
        reasons.append("last column (ML convention)")
    if any(kw in name_lower for kw in _TARGET_KEYWORDS):
        reasons.append("column name matches target keywords")
    non_null = df[best_col].dropna()
    unique_ratio = non_null.nunique() / len(non_null) if len(non_null) > 0 else 1.0
    if unique_ratio < 0.10:
        reasons.append(f"low cardinality ({non_null.nunique()} unique values → likely categorical target)")
    if not reasons:
        reasons.append("highest combined signal score")

    # Check ambiguity
    alternatives = []
    is_ambiguous = False
    for col in sorted_cols[1:3]:
        if scores[col] >= best_score * 0.85:
            is_ambiguous = True
            alternatives.append(col)

    return TargetDetectionResult(
        column=best_col,
        confidence=round(best_score, 3),
        reason=reasons,
        is_ambiguous=is_ambiguous,
        alternatives=alternatives,
    )
