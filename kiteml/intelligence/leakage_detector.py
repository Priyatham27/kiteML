"""
leakage_detector.py — Detect data leakage risks before training.

Flags columns that are suspiciously predictive of the target — indicating
possible leakage from future data, duplicate target encodings, or accidental
copies of the label.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class LeakageRisk:
    column: str
    correlation_with_target: float
    risk_level: str  # "low" | "high" | "critical"
    reason: str


@dataclass
class LeakageReport:
    has_leakage_risk: bool
    risks: List[LeakageRisk]
    critical_columns: List[str]
    recommendations: List[str]


def detect_leakage(
    df: pd.DataFrame,
    target: str,
    high_threshold: float = 0.90,
    critical_threshold: float = 0.98,
) -> LeakageReport:
    """
    Detect potential data leakage in the feature set.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including target column.
    target : str
        Name of the target column.
    high_threshold : float
        |correlation| above which a column is "high" risk. Default 0.90.
    critical_threshold : float
        |correlation| above which a column is "critical" risk. Default 0.98.

    Returns
    -------
    LeakageReport
    """
    risks: List[LeakageRisk] = []
    critical: List[str] = []
    recommendations: List[str] = []

    if target not in df.columns:
        return LeakageReport(False, [], [], ["Target column not found."])

    y = df[target]
    feature_cols = [c for c in df.columns if c != target]

    # ── Numeric correlation check ─────────────────────────────────────────
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0 and pd.api.types.is_numeric_dtype(y):
        corrs = df[num_cols].corrwith(y).abs().dropna()
        for col, corr_val in corrs.items():
            if corr_val >= high_threshold:
                level = "critical" if corr_val >= critical_threshold else "high"
                reason = f"Pearson |r| = {corr_val:.4f} with target — " "may be a target copy or future data."
                risks.append(
                    LeakageRisk(
                        column=col,
                        correlation_with_target=round(float(corr_val), 4),
                        risk_level=level,
                        reason=reason,
                    )
                )
                if level == "critical":
                    critical.append(col)

    # ── Exact duplicate detection (column == target after encoding) ───────
    for col in feature_cols:
        series = df[col]
        try:
            # Check if this column, when encoded numerically, is identical to y
            if pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(y):
                if series.equals(y.astype(series.dtype)):
                    risks.append(
                        LeakageRisk(
                            column=col,
                            correlation_with_target=1.0,
                            risk_level="critical",
                            reason="Column is an exact copy of the target.",
                        )
                    )
                    critical.append(col)
        except Exception:
            pass

    if critical:
        recommendations.append(f"🚨 Critical leakage risk in: {critical}. Remove these columns immediately.")
    if risks and not critical:
        recommendations.append("⚠️ High correlation with target detected. Verify these are valid features.")
    if not risks:
        recommendations.append("✅ No obvious leakage risks detected.")

    return LeakageReport(
        has_leakage_risk=len(risks) > 0,
        risks=risks,
        critical_columns=critical,
        recommendations=recommendations,
    )
