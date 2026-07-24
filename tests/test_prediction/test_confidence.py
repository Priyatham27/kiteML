"""
test_confidence.py — Unit tests for ConfidenceEngine (Story 5.6).
"""

import pytest
from sklearn.ensemble import RandomForestClassifier

from kiteml.prediction import ConfidenceEngine


def test_confidence_engine():
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit([[1, 2], [3, 4]], [0, 1])

    proba, conf = ConfidenceEngine().compute_confidence(clf, [[1, 2]])
    assert proba is not None
    assert conf is not None
    assert len(conf) == 1
    assert conf[0] >= 0.5
