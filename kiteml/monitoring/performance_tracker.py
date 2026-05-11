"""
performance_tracker.py — Track inference latency and throughput over time.

Provides sliding-window statistics on model inference performance,
including p50/p95/p99 latency, throughput, and degradation alerts.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class LatencySnapshot:
    """Aggregated latency snapshot for a window of requests."""
    window_size: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    max_ms: float
    min_ms: float
    requests_per_second: float
    window_duration_s: float


@dataclass
class PerformanceAlert:
    """A detected performance degradation."""
    alert_type: str    # "high_p99" | "low_throughput" | "latency_spike"
    value: float
    threshold: float
    message: str
    timestamp: str


class PerformanceTracker:
    """
    Sliding-window latency and throughput tracker.

    Parameters
    ----------
    window_size : int
        Number of most recent latency measurements to keep. Default 1000.
    p99_threshold_ms : float
        Alert when p99 latency exceeds this value. Default 500ms.
    min_throughput_rps : float
        Alert when throughput drops below this. Default 1.0 rps.
    """

    def __init__(
        self,
        window_size: int = 1_000,
        p99_threshold_ms: float = 500.0,
        min_throughput_rps: float = 1.0,
    ):
        self.window_size = window_size
        self.p99_threshold_ms = p99_threshold_ms
        self.min_throughput_rps = min_throughput_rps
        self._latencies: Deque[float] = deque(maxlen=window_size)
        self._timestamps: Deque[float] = deque(maxlen=window_size)
        self._total_requests: int = 0

    def record(self, latency_ms: float) -> None:
        """Record a single inference latency measurement."""
        self._latencies.append(latency_ms)
        self._timestamps.append(time.time())
        self._total_requests += 1

    def snapshot(self) -> Optional[LatencySnapshot]:
        """Compute current window statistics."""
        if len(self._latencies) < 2:
            return None

        arr = np.array(list(self._latencies))
        ts = list(self._timestamps)
        duration_s = ts[-1] - ts[0] if len(ts) > 1 else 1.0
        rps = len(arr) / duration_s if duration_s > 0 else 0.0

        return LatencySnapshot(
            window_size=len(arr),
            p50_ms=round(float(np.percentile(arr, 50)), 2),
            p95_ms=round(float(np.percentile(arr, 95)), 2),
            p99_ms=round(float(np.percentile(arr, 99)), 2),
            mean_ms=round(float(arr.mean()), 2),
            max_ms=round(float(arr.max()), 2),
            min_ms=round(float(arr.min()), 2),
            requests_per_second=round(rps, 2),
            window_duration_s=round(duration_s, 2),
        )

    def check_alerts(self) -> List[PerformanceAlert]:
        """Check for performance degradation alerts."""
        snap = self.snapshot()
        if snap is None:
            return []

        alerts: List[PerformanceAlert] = []
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if snap.p99_ms > self.p99_threshold_ms:
            alerts.append(PerformanceAlert(
                alert_type="high_p99",
                value=snap.p99_ms,
                threshold=self.p99_threshold_ms,
                message=f"p99 latency {snap.p99_ms:.0f}ms exceeds threshold {self.p99_threshold_ms:.0f}ms",
                timestamp=now,
            ))

        if snap.requests_per_second < self.min_throughput_rps and snap.window_size > 10:
            alerts.append(PerformanceAlert(
                alert_type="low_throughput",
                value=snap.requests_per_second,
                threshold=self.min_throughput_rps,
                message=f"Throughput {snap.requests_per_second:.2f} rps below min {self.min_throughput_rps:.2f} rps",
                timestamp=now,
            ))

        return alerts

    def print_report(self) -> None:
        """Print a formatted performance report."""
        snap = self.snapshot()
        if snap is None:
            print("⚠️  Not enough data for performance report.")
            return
        W = 50
        print(f"\n{'═'*W}")
        print("  ⚡  KiteML — Performance Tracker")
        print(f"{'═'*W}")
        print(f"  Window      : {snap.window_size} requests")
        print(f"  Duration    : {snap.window_duration_s:.1f}s")
        print(f"  Throughput  : {snap.requests_per_second:.1f} req/s")
        print(f"{'─'*W}")
        print(f"  p50 latency : {snap.p50_ms:.2f}ms")
        print(f"  p95 latency : {snap.p95_ms:.2f}ms")
        print(f"  p99 latency : {snap.p99_ms:.2f}ms")
        print(f"  mean        : {snap.mean_ms:.2f}ms")
        print(f"  min / max   : {snap.min_ms:.2f}ms / {snap.max_ms:.2f}ms")
        print(f"{'═'*W}")

    @property
    def total_requests(self) -> int:
        return self._total_requests
