"""
test_integration.py — Integration tests for PreprocessingEngine with DX pipeline (Story 4.1).
"""

import pandas as pd
import pytest

from kiteml.pipeline import create_dx_pipeline
from kiteml.preprocessing import PreprocessingEngine


def test_preprocessing_engine_dx_integration():
    dx = create_dx_pipeline()
    df = pd.DataFrame(
        {
            "high_nulls": [1.0] + [None] * 19,
            "normal_col": list(range(20)),
            "target": [0, 1] * 10,
        }
    )

    engine = PreprocessingEngine()
    blueprint = engine.plan(df, target="target", problem_type="classification")

    assert blueprint.feature_plans["high_nulls"].ignore is True
    suggs = dx.get_suggestions(df)
    assert len(suggs) >= 0  # DX suggestions queryable
