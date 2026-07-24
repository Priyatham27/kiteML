"""
test_integration.py — Integration tests for EvaluationEngine (Story 5.4).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from kiteml.evaluation import EvaluationEngine


def test_evaluation_engine_regression():
    X = pd.DataFrame({"x": np.linspace(1, 10, 10)})
    y = pd.Series(np.linspace(10, 100, 10))

    reg = LinearRegression()
    reg.fit(X, y)

    result = EvaluationEngine().evaluate(reg, X, y, problem_type="regression")
    assert result.report.task_type == "regression"
    assert "r2" in result.metrics
