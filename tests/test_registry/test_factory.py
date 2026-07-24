"""
test_factory.py — Unit tests for ModelFactory (Story 5.2).
"""

import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from kiteml.registry import model_registry


def test_model_factory_creation():
    clf = model_registry.create("RandomForestClassifier", params={"n_estimators": 10})
    assert isinstance(clf, RandomForestClassifier)
    assert clf.n_estimators == 10

    reg = model_registry.create("RandomForestRegressor")
    assert isinstance(reg, RandomForestRegressor)
