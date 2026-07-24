"""
test_profile.py — Integration tests for PreprocessingEngine with Epic 1 profiling (Story 4.1).
"""

import pandas as pd
import pytest

from kiteml.preprocessing import PreprocessingEngine


def test_preprocessing_engine_profile_integration():
    df = pd.DataFrame(
        {
            "id": list(range(20)),
            "price": [10.0, 20.0, 30.0, 40.0, 50.0] * 4,
            "category": ["A", "B", "C", "D", "E"] * 4,
            "date": pd.date_range("2025-01-01", periods=20, freq="D"),
            "target": [0, 1] * 10,
        }
    )

    engine = PreprocessingEngine()
    blueprint = engine.plan(df, target="target", problem_type="classification")

    assert blueprint.feature_count == 4
    assert "date" in blueprint.feature_plans
    date_plan = blueprint.feature_plans["date"]
    assert (
        date_plan.datetime_strategy.value == "extract_components" or date_plan.datetime_strategy == "extract_components"
    )
