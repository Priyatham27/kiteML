"""
test_benchmark.py — Unit tests for BenchmarkEngine (Story 5.4).
"""

import pytest
from sklearn.dummy import DummyClassifier

from kiteml.evaluation import BenchmarkEngine, BenchmarkMetrics


def test_benchmark_engine():
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit([[1], [2]], [0, 1])

    metrics = BenchmarkEngine().benchmark_model(clf, [[1], [2]])
    assert isinstance(metrics, BenchmarkMetrics)
    assert metrics.inference_latency_ms >= 0.0
