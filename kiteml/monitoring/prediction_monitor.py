"""
prediction_monitor.py — Track and analyze production prediction distributions.

Records every prediction made in production and surfaces:
  - Prediction distribution drift
  - Confidence score degradation
  - Anomalous prediction rates
  - Latency statistics
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class PredictionRecord:
    """A single recorded prediction."""
    timestamp: float
    prediction: Any
    confidence: Optional[float]   # max probability for classifiers
    latency_ms: float
    input_hash: Optional[str]


@dataclass
class PredictionStats:
    """Aggregated prediction statistics."""
    total_predictions: int
    unique_predictions: int
    prediction_distribution: Dict[str, float]
    avg_confidence: Optional[float]
    min_confidence: Optional[float]
    max_confidence: Optional[float]
    avg_latency_ms: float
    p99_latency_ms: float
    time_window_s: float


class PredictionMonitor:
    """
    Records and analyzes production predictions.

    Parameters
    ----------
    max_history : int
        Maximum number of predictions to keep in memory. Default 10000.
    """

    def __init__(self, max_history: int = 10_000):
        self.max_history = max_history
        self._records: List[PredictionRecord] = []

    def record(
        self,
        prediction: Any,
        latency_ms: float = 0.0,
        confidence: Optional[float] = None,
        input_data: Optional[Any] = None,
    ) -> None:
        """Record a single prediction."""
        input_hash = None
        if input_data is not None:
            try:
                import hashlib
                input_hash = hashlib.md5(str(input_data).encode()).hexdigest()[:8]
            except Exception:
                pass

        record = PredictionRecord(
            timestamp=time.time(),
            prediction=prediction,
            confidence=confidence,
            latency_ms=latency_ms,
            input_hash=input_hash,
        )
        self._records.append(record)
        if len(self._records) > self.max_history:
            self._records = self._records[-self.max_history:]

    def stats(self, last_n: Optional[int] = None) -> PredictionStats:
        """Compute aggregate statistics over recorded predictions."""
        records = self._records[-last_n:] if last_n else self._records
        if not records:
            return PredictionStats(
                total_predictions=0, unique_predictions=0,
                prediction_distribution={}, avg_confidence=None,
                min_confidence=None, max_confidence=None,
                avg_latency_ms=0.0, p99_latency_ms=0.0, time_window_s=0.0,
            )

        preds = [str(r.prediction) for r in records]
        latencies = [r.latency_ms for r in records]
        confidences = [r.confidence for r in records if r.confidence is not None]

        counts = defaultdict(int)
        for p in preds:
            counts[p] += 1
        total = len(records)
        dist = {k: round(v / total, 4) for k, v in counts.items()}

        time_window = records[-1].timestamp - records[0].timestamp if len(records) > 1 else 0

        return PredictionStats(
            total_predictions=total,
            unique_predictions=len(counts),
            prediction_distribution=dist,
            avg_confidence=round(float(np.mean(confidences)), 4) if confidences else None,
            min_confidence=round(float(np.min(confidences)), 4) if confidences else None,
            max_confidence=round(float(np.max(confidences)), 4) if confidences else None,
            avg_latency_ms=round(float(np.mean(latencies)), 2),
            p99_latency_ms=round(float(np.percentile(latencies, 99)), 2),
            time_window_s=round(time_window, 1),
        )

    def check_anomalies(
        self,
        low_confidence_threshold: float = 0.6,
        high_latency_threshold_ms: float = 500.0,
    ) -> List[str]:
        """Return list of detected anomaly warnings."""
        warnings: List[str] = []
        stats = self.stats()
        if stats.avg_confidence is not None and stats.avg_confidence < low_confidence_threshold:
            warnings.append(
                f"⚠️ Low average confidence: {stats.avg_confidence:.2%} "
                f"(threshold: {low_confidence_threshold:.0%})"
            )
        if stats.p99_latency_ms > high_latency_threshold_ms:
            warnings.append(
                f"⚠️ High p99 latency: {stats.p99_latency_ms:.0f}ms "
                f"(threshold: {high_latency_threshold_ms:.0f}ms)"
            )
        return warnings

    def reset(self) -> None:
        """Clear all recorded predictions."""
        self._records.clear()
