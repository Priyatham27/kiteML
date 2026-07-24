"""
test_engine.py — Unit tests for PreprocessingEngine (Story 4.1).
"""

import pandas as pd
import pytest

from kiteml.preprocessing import (
    EncodingStrategy,
    MissingStrategy,
    PreprocessingEngine,
    ScalingStrategy,
)


def test_preprocessing_engine_planning():
    df = pd.DataFrame(
        {
            "age": [25, 30, None, 40, 50, 60, 70, 80, 90, 100],
            "city": ["NY", "LA", "NY", "SF", "LA", "NY", "SF", "LA", "NY", "SF"],
            "target": [0, 1] * 5,
        }
    )

    engine = PreprocessingEngine()
    blueprint = engine.plan(df, target="target", problem_type="classification")

    assert blueprint.feature_count == 2
    assert "age" in blueprint.feature_plans
    assert "city" in blueprint.feature_plans

    age_plan = blueprint.feature_plans["age"]
    assert age_plan.missing_strategy == MissingStrategy.MEDIAN
    assert age_plan.scaling_strategy in (ScalingStrategy.STANDARD, ScalingStrategy.ROBUST)

    city_plan = blueprint.feature_plans["city"]
    assert city_plan.encoding_strategy == EncodingStrategy.ONE_HOT
    assert city_plan.scaling_strategy == ScalingStrategy.NONE


def test_preprocessing_engine_empty_dataframe():
    engine = PreprocessingEngine()
    blueprint = engine.plan(pd.DataFrame())
    assert blueprint.feature_count == 0
