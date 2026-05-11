"""
problem_inference.py — Advanced problem type inference for KiteML.

Goes beyond a simple numeric/categorical check to distinguish:
  - binary classification
  - multi-class classification
  - regression
  - (future) multi-label, ordinal regression

Uses entropy, unique value ratio, and distribution analysis.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ProblemInferenceResult:
    """Result of advanced problem type inference."""

    problem_type: str  # "classification" | "regression"
    subtype: str  # "binary" | "multiclass" | "continuous" | "discrete_regression"
    confidence: float  # 0–1
    evidence: list[str]  # human-readable justification
    n_classes: Optional[int] = None


def infer_problem_type_advanced(
    target: pd.Series,
) -> ProblemInferenceResult:
    """
    Infer the ML problem type from the target series alone.

    Parameters
    ----------
    target : pd.Series
        The target column (labels or values).

    Returns
    -------
    ProblemInferenceResult
    """
    non_null = target.dropna()
    n_total = len(non_null)
    n_unique = int(non_null.nunique())
    unique_ratio = n_unique / n_total if n_total > 0 else 0.0
    evidence: list[str] = []

    # ── String target → always classification ─────────────────────────────
    if pd.api.types.is_object_dtype(target) or pd.api.types.is_string_dtype(target):
        evidence.append("string dtype → classification")
        subtype = "binary" if n_unique == 2 else "multiclass"
        return ProblemInferenceResult(
            problem_type="classification",
            subtype=subtype,
            confidence=0.98,
            evidence=evidence,
            n_classes=n_unique,
        )

    # ── Boolean → binary classification ───────────────────────────────────
    if pd.api.types.is_bool_dtype(target):
        evidence.append("bool dtype → binary classification")
        return ProblemInferenceResult(
            problem_type="classification",
            subtype="binary",
            confidence=0.99,
            evidence=evidence,
            n_classes=2,
        )

    # ── Numeric target ────────────────────────────────────────────────────
    # Entropy-based: few unique values relative to total → classification
    # Signal 1: unique ratio
    if unique_ratio < 0.02 or n_unique <= 20:
        evidence.append(f"{n_unique} unique values (ratio={unique_ratio:.4f}) → low cardinality")

        # Check if values are integers only
        is_integer_valued = np.array_equal(non_null.values, np.floor(non_null.values))
        if is_integer_valued:
            evidence.append("integer-only values")

        if n_unique == 2:
            subtype = "binary"
            confidence = 0.95
        elif n_unique <= 20:
            subtype = "multiclass"
            confidence = 0.88
        else:
            subtype = "multiclass"
            confidence = 0.75

        return ProblemInferenceResult(
            problem_type="classification",
            subtype=subtype,
            confidence=confidence,
            evidence=evidence,
            n_classes=n_unique,
        )

    # Signal 2: float dtype or high unique ratio → regression
    if pd.api.types.is_float_dtype(target):
        evidence.append("float dtype")
    if unique_ratio > 0.10:
        evidence.append(f"high unique ratio ({unique_ratio:.4f})")

    # Signal 3: distribution continuity (std relative to range)
    val_range = float(non_null.max() - non_null.min())
    if val_range > 0:
        std = float(non_null.std())
        smoothness = std / val_range
        if smoothness > 0.05:
            evidence.append(f"smooth distribution (std/range={smoothness:.3f})")

    # Check if discrete regression (integers but many values)
    is_integer_valued = np.array_equal(non_null.values, np.floor(non_null.values))
    if is_integer_valued and n_unique > 20:
        evidence.append("integer values but high cardinality → discrete regression")
        return ProblemInferenceResult(
            problem_type="regression",
            subtype="discrete_regression",
            confidence=0.80,
            evidence=evidence,
        )

    evidence.append("continuous numeric target → regression")
    return ProblemInferenceResult(
        problem_type="regression",
        subtype="continuous",
        confidence=0.92,
        evidence=evidence,
    )
