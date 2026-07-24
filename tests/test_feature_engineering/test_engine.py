"""
test_engine.py — Unit tests for FeatureEngineeringEngine (Story 4.2).
"""

import pandas as pd
import pytest

from kiteml.feature_engineering import FeatureEngineeringEngine


def test_feature_engineering_engine_planning():
    df = pd.DataFrame(
        {
            "price": [10.0, 20.0, 30.0, 40.0, 50.0] * 4,
            "quantity": [1, 2, 3, 4, 5] * 4,
            "date": pd.date_range("2025-01-01", periods=20, freq="D"),
            "target": [0, 1] * 10,
        }
    )

    engine = FeatureEngineeringEngine()
    blueprint = engine.plan(df, target="target", problem_type="classification")

    assert blueprint.feature_count > 0
    assert "date_year" in blueprint.feature_plans
    assert "price_x_quantity" in blueprint.feature_plans


def test_feature_engineering_engine_empty_dataframe():
    engine = FeatureEngineeringEngine()
    blueprint = engine.plan(pd.DataFrame())
    assert blueprint.feature_count == 0
