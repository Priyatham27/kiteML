"""
test_engine.py — Unit tests for EvaluationEngine (Story 5.4).
"""

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from kiteml.evaluation import EvaluationEngine, EvaluationResult


def test_evaluation_engine_evaluate():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    y = pd.Series([0, 1, 0, 1])

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    res = EvaluationEngine().evaluate(clf, X, y, problem_type="classification")
    assert isinstance(res, EvaluationResult)
    assert res.composite_score > 0.0
    assert "📈 KiteML Model Evaluation Report" in res.report.summary()
