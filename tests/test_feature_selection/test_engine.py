"""
test_engine.py — Unit tests for FeatureSelectionEngine (Story 4.3).
"""

import pandas as pd
import pytest

from kiteml.feature_selection import FeatureSelectionEngine, SelectionDecision


def test_feature_selection_engine_planning():
    df = pd.DataFrame(
        {
            "id": list(range(20)),
            "constant_col": [1] * 20,
            "feature1": list(range(20)),
            "target": [0, 1] * 10,
        }
    )

    engine = FeatureSelectionEngine()
    blueprint = engine.plan(df, target="target", problem_type="classification")

    assert blueprint.total_features == 3
    assert "constant_col" in blueprint.removed_features
    assert "feature1" in blueprint.selected_features


def test_feature_selection_engine_protected_features():
    df = pd.DataFrame(
        {
            "constant_col": [1] * 20,
            "feature1": list(range(20)),
            "target": [0, 1] * 10,
        }
    )

    engine = FeatureSelectionEngine()
    blueprint = engine.plan(df, target="target", keep_features=["constant_col"])

    assert "constant_col" in blueprint.selected_features
    assert blueprint.feature_scores["constant_col"].is_protected is True
