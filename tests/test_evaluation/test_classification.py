"""
test_classification.py — Unit tests for classification evaluation (Story 5.4).
"""

import pytest

from kiteml.evaluation import evaluate_classification_metrics


def test_classification_metrics():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]

    metrics = evaluate_classification_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["accuracy"] == 0.75
