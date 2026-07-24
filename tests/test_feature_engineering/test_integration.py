"""
test_integration.py — Integration tests for FeatureEngineeringEngine (Story 4.2).
"""

import pandas as pd
import pytest

from kiteml.feature_engineering import FeatureEngineeringEngine
from kiteml.preprocessing import PreprocessingEngine


def test_feature_engineering_preprocessing_integration():
    df = pd.DataFrame(
        {
            "price": [10.0, 20.0, 30.0, 40.0, 50.0] * 4,
            "quantity": [1, 2, 3, 4, 5] * 4,
            "category": ["A", "B", "C", "D", "E"] * 4,
            "target": [0, 1] * 10,
        }
    )

    prep_engine = PreprocessingEngine()
    prep_bp = prep_engine.plan(df, target="target", problem_type="classification")

    fe_engine = FeatureEngineeringEngine()
    fe_bp = fe_engine.plan(
        df,
        preprocessing_blueprint=prep_bp,
        target="target",
        problem_type="classification",
    )

    assert fe_bp.feature_count > 0
    assert "price_x_quantity" in fe_bp.feature_plans
