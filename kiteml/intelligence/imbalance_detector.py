"""
imbalance_detector.py — Class imbalance detection for classification targets.

Measures target distribution and flags severe imbalance that can lead to
misleading accuracy metrics and biased models.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ImbalanceReport:
    """Result of class imbalance detection."""

    is_imbalanced: bool
    severity: str  # "none" | "mild" | "moderate" | "severe" | "extreme"
    imbalance_ratio: float  # max_class / min_class count
    class_distribution: dict[Any, float]  # class → fraction
    majority_class: Any
    minority_class: Any
    majority_ratio: float
    minority_ratio: float
    recommendations: list[str]


def detect_imbalance(
    target: pd.Series,
    mild_threshold: float = 1.5,
    moderate_threshold: float = 3.0,
    severe_threshold: float = 10.0,
    extreme_threshold: float = 20.0,
) -> ImbalanceReport:
    """
    Analyze class distribution of a classification target.

    Parameters
    ----------
    target : pd.Series
    mild_threshold : float
        Ratio above which imbalance is "mild". Default 1.5.
    moderate_threshold : float
        Ratio above which imbalance is "moderate". Default 3.
    severe_threshold : float
        Ratio above which imbalance is "severe". Default 10.
    extreme_threshold : float
        Ratio above which imbalance is "extreme". Default 20.

    Returns
    -------
    ImbalanceReport
    """
    vc = target.dropna().value_counts()
    n_total = vc.sum()

    if n_total == 0 or len(vc) < 2:
        return ImbalanceReport(
            is_imbalanced=False,
            severity="none",
            imbalance_ratio=1.0,
            class_distribution={},
            majority_class=None,
            minority_class=None,
            majority_ratio=1.0,
            minority_ratio=1.0,
            recommendations=["Only one class detected — check target column."],
        )

    class_dist = {k: round(v / n_total, 4) for k, v in vc.items()}
    majority_class = vc.idxmax()
    minority_class = vc.idxmin()
    majority_count = int(vc.max())
    minority_count = int(vc.min())
    ratio = majority_count / minority_count if minority_count > 0 else float("inf")

    if ratio < mild_threshold:
        severity = "none"
        is_imbalanced = False
    elif ratio < moderate_threshold:
        severity = "mild"
        is_imbalanced = True
    elif ratio < severe_threshold:
        severity = "moderate"
        is_imbalanced = True
    elif ratio < extreme_threshold:
        severity = "severe"
        is_imbalanced = True
    else:
        severity = "extreme"
        is_imbalanced = True

    recommendations: list[str] = []
    if severity in ("severe", "extreme"):
        recommendations += [
            f"⚠️ Severe imbalance: {majority_class}={class_dist[majority_class]:.1%}, "
            f"{minority_class}={class_dist[minority_class]:.1%}",
            "Use stratified train/test splitting (already applied by KiteML).",
            "Consider SMOTE oversampling or class_weight='balanced'.",
            "Prefer F1/AUC-ROC over accuracy as your primary metric.",
        ]
    elif severity == "moderate":
        recommendations += [
            "Moderate imbalance detected.",
            "Consider using class_weight='balanced' in your models.",
        ]
    elif severity == "mild":
        recommendations.append("Mild imbalance — monitor F1 score alongside accuracy.")

    return ImbalanceReport(
        is_imbalanced=is_imbalanced,
        severity=severity,
        imbalance_ratio=round(ratio, 2),
        class_distribution=class_dist,
        majority_class=majority_class,
        minority_class=minority_class,
        majority_ratio=round(majority_count / n_total, 4),
        minority_ratio=round(minority_count / n_total, 4),
        recommendations=recommendations,
    )
