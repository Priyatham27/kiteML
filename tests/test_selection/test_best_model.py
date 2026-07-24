"""
test_best_model.py — Unit tests for BestModel container (Story 5.5).
"""

import pytest
from sklearn.dummy import DummyClassifier

from kiteml.selection import BestModel


def test_best_model_predict():
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit([[1], [2]], [0, 1])

    bm = BestModel(
        model=dummy,
        model_name="DummyClassifier",
        composite_score=85.0,
        explanation="Baseline model",
    )

    preds = bm.predict([[1]])
    assert len(preds) == 1
    assert "🏆 KiteML Optimal Selected Model" in bm.summary()
