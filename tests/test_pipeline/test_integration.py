"""
test_integration.py — Integration tests for TransformationPipeline with pre-calculated blueprints (Story 4.4).
"""

import pandas as pd
import pytest

from kiteml.feature_engineering import FeatureEngineeringEngine
from kiteml.feature_selection import FeatureSelectionEngine
from kiteml.pipeline import TransformationPipeline
from kiteml.preprocessing import PreprocessingEngine


def test_pipeline_with_precalculated_blueprints():
    df = pd.DataFrame(
        {
            "price": [10.0, 20.0, 30.0, 40.0, 50.0] * 4,
            "quantity": [1, 2, 3, 4, 5] * 4,
            "target": [0, 1] * 10,
        }
    )

    prep_engine = PreprocessingEngine()
    prep_bp = prep_engine.plan(df, target="target")

    fe_engine = FeatureEngineeringEngine()
    fe_bp = fe_engine.plan(df, preprocessing_blueprint=prep_bp, target="target")

    fs_engine = FeatureSelectionEngine()
    fs_bp = fs_engine.plan(df, preprocessing_blueprint=prep_bp, feature_engineering_blueprint=fe_bp, target="target")

    pipeline = TransformationPipeline(
        preprocessing_blueprint=prep_bp,
        engineering_blueprint=fe_bp,
        selection_blueprint=fs_bp,
    )

    transformed = pipeline.fit_transform(df, target="target")
    assert isinstance(transformed, pd.DataFrame)
    assert not transformed.empty
