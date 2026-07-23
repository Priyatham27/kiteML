"""
test_pipeline.py — Unit tests for ValidationPipeline (Story 2.6).
"""

import pandas as pd
import pytest

from kiteml.validation.pipeline import ValidationPipeline


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_2": ["A", "B", "A", "B", "A"],
            "target": [0, 1, 0, 1, 0],
        }
    )


def test_pipeline_execution_order_and_aggregation(sample_df):
    pipeline = ValidationPipeline()
    summary = pipeline.validate(sample_df, target="target", problem_type="classification")

    assert summary.passed is True
    assert summary.ready_for_training is True
    assert summary.health_score >= 90
    assert summary.health_grade in ("A+", "A")
    assert "DatasetValidator" in summary.validator_results
    assert "TargetValidator" in summary.validator_results
    assert "SchemaValidator" in summary.validator_results
    assert "QualityValidator" in summary.validator_results
