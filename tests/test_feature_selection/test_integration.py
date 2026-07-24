"""
test_integration.py — Integration tests for FeatureSelectionEngine (Story 4.3).
"""

import pandas as pd
import pytest

from kiteml.feature_engineering import FeatureEngineeringEngine
from kiteml.feature_selection import FeatureSelectionEngine
from kiteml.preprocessing import PreprocessingEngine


def test_full_pipeline_planning_integration():
    df = pd.DataFrame(
        {
            "id": list(range(20)),
            "constant_col": [100] * 20,
            "price": [10.0, 20.0, 30.0, 40.0, 50.0] * 4,
            "quantity": [1, 2, 3, 4, 5] * 4,
            "target": [0, 1] * 10,
        }
    )

    prep_engine = PreprocessingEngine()
    prep_bp = prep_engine.plan(df, target="target", problem_type="classification")

    fe_engine = FeatureEngineeringEngine()
    fe_bp = fe_engine.plan(df, preprocessing_blueprint=prep_bp, target="target", problem_type="classification")

    fs_engine = FeatureSelectionEngine()
    fs_bp = fs_engine.plan(
        df,
        preprocessing_blueprint=prep_bp,
        feature_engineering_blueprint=fe_bp,
        target="target",
        problem_type="classification",
    )

    assert fs_bp.total_features == 4
    assert "constant_col" in fs_bp.removed_features
    assert "price" in fs_bp.selected_features
