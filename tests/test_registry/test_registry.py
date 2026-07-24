"""
test_registry.py — Unit tests for ModelRegistry (Story 5.2).
"""

import pytest

from kiteml.registry import ModelRegistry, model_registry


def test_model_registry_list_and_search():
    reg = ModelRegistry()

    regr_models = reg.list_models(task="regression")
    assert "LinearRegression" in regr_models
    assert "RandomForestRegressor" in regr_models

    cls_models = reg.list_models(task="binary_classification")
    assert "LogisticRegression" in cls_models
    assert "RandomForestClassifier" in cls_models

    tree_models = reg.search(tags=["tree"])
    assert "RandomForestClassifier" in tree_models
