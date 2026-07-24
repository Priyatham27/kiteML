"""
test_engine.py — Unit tests for ModelSelectionEngine (Story 5.5).
"""

import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

from kiteml.selection import BestModel, ModelSelectionEngine, SelectionReport


def test_model_selection_engine():
    clf1 = DummyClassifier(strategy="most_frequent")
    clf1.fit([[1]], [0])
    clf2 = DummyClassifier(strategy="stratified")
    clf2.fit([[1]], [0])

    candidates = [
        {"name": "Dummy1", "model": clf1, "composite_score": 70.0, "benchmark": {"inference_latency_ms": 1.0}},
        {"name": "Dummy2", "model": clf2, "composite_score": 90.0, "benchmark": {"inference_latency_ms": 0.5}},
    ]

    engine = ModelSelectionEngine()
    best = engine.select(candidates, policy="balanced")

    assert isinstance(best, BestModel)
    assert best.model_name == "Dummy2"
    assert best.composite_score > 0.0

    report = engine.generate_report(candidates, policy="balanced")
    assert isinstance(report, SelectionReport)
    assert "👑 KiteML Model Leaderboard" in report.summary()
