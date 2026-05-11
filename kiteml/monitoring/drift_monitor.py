"""
drift_monitor.py — Production data drift detection for KiteML.

Detects feature drift, target drift, and prediction drift using:
  - PSI (Population Stability Index)
  - KS statistic (Kolmogorov-Smirnov)
  - Mean/std shift

PSI Guide:
  < 0.1  → No significant change
  0.1–0.2 → Moderate change (monitor)
  > 0.2  → Significant change (retrain)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureDriftResult:
    """Drift result for one feature."""

    feature: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    mean_shift: Optional[float]
    std_shift: Optional[float]
    drift_detected: bool
    severity: str  # "none" | "moderate" | "high"


@dataclass
class DriftReport:
    """Complete data drift analysis."""

    reference_rows: int
    current_rows: int
    drifted_features: List[str]
    feature_results: Dict[str, FeatureDriftResult]
    overall_psi: float
    drift_detected: bool
    severity: str
    recommendations: List[str]

    def summary(self) -> str:
        icon = "🚨" if self.drift_detected else "✅"
        return (
            f"{icon} Drift {'DETECTED' if self.drift_detected else 'NOT DETECTED'} "
            f"| Overall PSI: {self.overall_psi:.4f} | "
            f"Drifted features: {len(self.drifted_features)}/{len(self.feature_results)}"
        )


def _compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Compute Population Stability Index between two numeric distributions."""
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    if min_val == max_val:
        return 0.0
    bins = np.linspace(min_val, max_val, n_bins + 1)
    ref_hist, _ = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current, bins=bins)

    ref_pct = (ref_hist + epsilon) / (len(reference) + epsilon * n_bins)
    cur_pct = (cur_hist + epsilon) / (len(current) + epsilon * n_bins)
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct + epsilon)))
    return round(abs(psi), 4)


def _compute_ks(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Compute KS statistic and p-value."""
    try:
        from scipy.stats import ks_2samp

        stat, p = ks_2samp(a, b)
        return round(float(stat), 4), round(float(p), 4)
    except ImportError:
        # Simple manual KS without scipy
        a_sorted = np.sort(a)
        b_sorted = np.sort(b)
        combined = np.sort(np.concatenate([a_sorted, b_sorted]))
        cdf_a = np.searchsorted(a_sorted, combined, side="right") / len(a_sorted)
        cdf_b = np.searchsorted(b_sorted, combined, side="right") / len(b_sorted)
        stat = float(np.max(np.abs(cdf_a - cdf_b)))
        return round(stat, 4), -1.0  # p unknown without scipy


def check_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    psi_threshold_moderate: float = 0.1,
    psi_threshold_high: float = 0.2,
    ks_threshold: float = 0.05,
) -> DriftReport:
    """
    Detect data drift between reference (training) and current (production) data.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Training / baseline dataset.
    current_df : pd.DataFrame
        Current production dataset.
    feature_names : list of str, optional
        Columns to monitor. Defaults to all numeric columns in both.
    psi_threshold_moderate : float
        PSI above which drift is "moderate". Default 0.1.
    psi_threshold_high : float
        PSI above which drift is "high". Default 0.2.
    ks_threshold : float
        KS p-value below which drift is significant. Default 0.05.

    Returns
    -------
    DriftReport
    """
    numeric_ref = reference_df.select_dtypes(include=[np.number])
    numeric_cur = current_df.select_dtypes(include=[np.number])
    common_cols = list(set(numeric_ref.columns) & set(numeric_cur.columns))
    if feature_names:
        common_cols = [f for f in feature_names if f in common_cols]

    feature_results: Dict[str, FeatureDriftResult] = {}
    all_psis: List[float] = []
    drifted: List[str] = []

    for feat in common_cols:
        ref = numeric_ref[feat].dropna().values
        cur = numeric_cur[feat].dropna().values
        if len(ref) < 10 or len(cur) < 10:
            continue

        psi = _compute_psi(ref, cur)
        ks_stat, ks_p = _compute_ks(ref, cur)
        mean_shift = round(float(cur.mean() - ref.mean()), 4)
        std_shift = round(float(cur.std() - ref.std()), 4)

        if psi >= psi_threshold_high:
            severity = "high"
            drift_detected = True
        elif psi >= psi_threshold_moderate or (ks_p != -1 and ks_p < ks_threshold):
            severity = "moderate"
            drift_detected = True
        else:
            severity = "none"
            drift_detected = False

        all_psis.append(psi)
        if drift_detected:
            drifted.append(feat)

        feature_results[feat] = FeatureDriftResult(
            feature=feat,
            psi=psi,
            ks_statistic=ks_stat,
            ks_p_value=ks_p,
            mean_shift=mean_shift,
            std_shift=std_shift,
            drift_detected=drift_detected,
            severity=severity,
        )

    overall_psi = round(float(np.mean(all_psis)) if all_psis else 0.0, 4)
    overall_drift = len(drifted) > 0

    if overall_psi >= psi_threshold_high:
        overall_severity = "high"
    elif overall_psi >= psi_threshold_moderate:
        overall_severity = "moderate"
    else:
        overall_severity = "none"

    recommendations: List[str] = []
    if overall_severity == "high":
        recommendations.append(f"🚨 High drift detected (PSI={overall_psi:.3f}). Consider retraining.")
    elif overall_severity == "moderate":
        recommendations.append(f"⚠️ Moderate drift detected (PSI={overall_psi:.3f}). Monitor closely.")
    else:
        recommendations.append(f"✅ No significant drift (PSI={overall_psi:.3f}).")

    if drifted:
        top = sorted(drifted, key=lambda f: feature_results[f].psi, reverse=True)[:3]
        recommendations.append(f"Most drifted features: {top}")

    return DriftReport(
        reference_rows=len(reference_df),
        current_rows=len(current_df),
        drifted_features=drifted,
        feature_results=feature_results,
        overall_psi=overall_psi,
        drift_detected=overall_drift,
        severity=overall_severity,
        recommendations=recommendations,
    )
