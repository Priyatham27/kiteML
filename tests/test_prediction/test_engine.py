"""
test_engine.py — Unit tests for PredictionEngine (Story 5.6).
"""

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from kiteml.prediction import PredictionEngine, PredictionResult


def test_prediction_engine():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    y = pd.Series([0, 1, 0, 1])

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    engine = PredictionEngine()
    result = engine.predict(model=clf, dataframe=X)

    assert isinstance(result, PredictionResult)
    assert len(result) == 4
    assert result.confidence is not None
    assert "🔮 KiteML Prediction Execution Summary" in result.report.summary()
