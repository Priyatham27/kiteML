"""
anomaly_monitor.py — Detect anomalous production inputs for KiteML.

Uses IQR and Z-score methods on training statistics to flag
production rows with extreme or suspicious feature values.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FeatureStats:
    """Baseline statistics for one feature (from training data)."""

    name: str
    mean: float
    std: float
    q1: float
    q3: float
    iqr: float
    min_val: float
    max_val: float


@dataclass
class AnomalyResult:
    """Result of anomaly detection on a batch of inputs."""

    n_rows: int
    n_anomalous: int
    anomaly_ratio: float
    anomalous_rows: list[int]  # row indices
    anomalous_features: dict[int, list[str]]  # row_idx → flagged features
    feature_violation_counts: dict[str, int]
    has_anomalies: bool

    def summary(self) -> str:
        pct = self.anomaly_ratio * 100
        return (
            f"Anomaly check: {self.n_anomalous}/{self.n_rows} rows flagged "
            f"({pct:.1f}%) | "
            f"Top features: {list(self.feature_violation_counts.keys())[:3]}"
        )


class AnomalyMonitor:
    """
    Flags production inputs that deviate from training-time distributions.

    Parameters
    ----------
    method : str
        ``'iqr'`` (default) or ``'zscore'``.
    iqr_multiplier : float
        IQR fence multiplier. Default 3.0 (less sensitive than 1.5).
    zscore_threshold : float
        Absolute Z-score above which a value is anomalous. Default 4.0.
    """

    def __init__(
        self,
        method: str = "iqr",
        iqr_multiplier: float = 3.0,
        zscore_threshold: float = 4.0,
    ):
        self.method = method
        self.iqr_multiplier = iqr_multiplier
        self.zscore_threshold = zscore_threshold
        self._stats: dict[str, FeatureStats] = {}

    def fit(self, df: pd.DataFrame) -> "AnomalyMonitor":
        """
        Compute baseline statistics from training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training / reference DataFrame.
        """
        for col in df.select_dtypes(include=[np.number]).columns:
            vals = df[col].dropna().values
            if len(vals) == 0:
                continue
            q1, q3 = np.percentile(vals, [25, 75])
            self._stats[col] = FeatureStats(
                name=col,
                mean=float(vals.mean()),
                std=float(vals.std()),
                q1=float(q1),
                q3=float(q3),
                iqr=float(q3 - q1),
                min_val=float(vals.min()),
                max_val=float(vals.max()),
            )
        return self

    def check(self, df: pd.DataFrame) -> AnomalyResult:
        """
        Detect anomalous rows in production input data.

        Parameters
        ----------
        df : pd.DataFrame
            Production input data.

        Returns
        -------
        AnomalyResult
        """
        if not self._stats:
            raise RuntimeError("Call fit() with training data before check().")

        n_rows = len(df)
        row_flags: dict[int, list[str]] = {}
        feature_violations: dict[str, int] = {}

        for col, stats in self._stats.items():
            if col not in df.columns:
                continue

            vals = df[col].values

            if self.method == "iqr":
                lower = stats.q1 - self.iqr_multiplier * stats.iqr
                upper = stats.q3 + self.iqr_multiplier * stats.iqr
                flagged_mask = (vals < lower) | (vals > upper)
            else:  # zscore
                if stats.std == 0:
                    continue
                z = np.abs((vals - stats.mean) / stats.std)
                flagged_mask = z > self.zscore_threshold

            flagged_rows = list(np.where(flagged_mask)[0])
            if flagged_rows:
                feature_violations[col] = len(flagged_rows)
                for row_idx in flagged_rows:
                    row_flags.setdefault(int(row_idx), []).append(col)

        anomalous_rows = sorted(row_flags.keys())
        return AnomalyResult(
            n_rows=n_rows,
            n_anomalous=len(anomalous_rows),
            anomaly_ratio=round(len(anomalous_rows) / n_rows, 4) if n_rows > 0 else 0.0,
            anomalous_rows=anomalous_rows,
            anomalous_features=row_flags,
            feature_violation_counts=feature_violations,
            has_anomalies=len(anomalous_rows) > 0,
        )

    @classmethod
    def from_result(cls, result: Any, training_df: pd.DataFrame, **kwargs) -> "AnomalyMonitor":
        """Convenience: fit monitor from a KiteML Result + training data."""
        monitor = cls(**kwargs)
        feature_cols = [c for c in (result.feature_names or []) if c in training_df.columns]
        monitor.fit(training_df[feature_cols] if feature_cols else training_df)
        return monitor
