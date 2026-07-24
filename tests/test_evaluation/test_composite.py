"""
test_composite.py — Unit tests for CompositeScorer (Story 5.4 Flagship Feature).
"""

import pytest

from kiteml.evaluation import BenchmarkMetrics, CompositeScorer


def test_composite_scorer():
    scorer = CompositeScorer()
    bm = BenchmarkMetrics(inference_latency_ms=1.0)
    score = scorer.calculate_composite_score({"f1": 0.9}, bm, task_type="classification")

    assert 0.0 <= score <= 100.0
    assert score > 70.0
