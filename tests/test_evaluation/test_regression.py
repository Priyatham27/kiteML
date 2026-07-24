"""
test_regression.py — Unit tests for regression evaluation (Story 5.4).
"""

import pytest

from kiteml.evaluation import evaluate_regression_metrics


def test_regression_metrics():
    y_true = [10.0, 20.0, 30.0]
    y_pred = [10.5, 19.5, 30.0]

    metrics = evaluate_regression_metrics(y_true, y_pred)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["mae"] > 0.0
