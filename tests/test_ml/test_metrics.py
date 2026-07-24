"""
test_metrics.py — Unit tests for UnifiedMetricsEngine (Story 5.8).
"""

import time

import pytest

from kiteml.ml import UnifiedMetricsEngine


def test_unified_metrics_engine():
    engine = UnifiedMetricsEngine()
    engine.start_workflow()
    time.sleep(0.01)
    engine.record_stage_time("train", 0.05)

    summary = engine.get_summary()
    assert summary["total_execution_time_sec"] > 0
    assert summary["stage_timings_sec"]["train"] == 0.05
