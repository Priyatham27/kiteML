"""
test_integration.py — Integration tests for ModelSelectionEngine (Story 5.5).
"""

import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from kiteml.selection import ModelSelectionEngine


def test_model_selection_integration():
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit([[1, 2], [3, 4]], [0, 1])

    candidates = [
        {
            "name": "RandomForestClassifier",
            "model": rf,
            "composite_score": 94.5,
            "metrics": {"accuracy": 1.0, "f1": 1.0},
            "benchmark": {"inference_latency_ms": 2.1, "model_size_kb": 50.0},
        }
    ]

    engine = ModelSelectionEngine()
    best = engine.select(candidates, policy="accuracy")

    assert best.model_name == "RandomForestClassifier"
    assert best.model == rf
