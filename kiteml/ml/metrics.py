"""
metrics.py — UnifiedMetricsEngine for tracking execution latency and resource metrics in KiteML.
"""

import time
from typing import Any


class UnifiedMetricsEngine:
    """
    Tracks stage timing and workflow resource metrics.
    """

    def __init__(self) -> None:
        self.stage_timings: dict[str, float] = {}
        self.total_start_time: float = 0.0

    def start_workflow(self) -> None:
        """Start total workflow timer."""
        self.total_start_time = time.time()

    def record_stage_time(self, stage_name: str, duration_sec: float) -> None:
        """Record stage execution duration."""
        self.stage_timings[stage_name] = round(duration_sec, 4)

    def get_summary(self) -> dict[str, Any]:
        """Get metric summary dictionary."""
        total_dur = time.time() - self.total_start_time if self.total_start_time > 0 else 0.0
        return {
            "total_execution_time_sec": round(total_dur, 4),
            "stage_timings_sec": self.stage_timings,
        }
