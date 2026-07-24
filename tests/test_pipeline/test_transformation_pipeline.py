"""
test_transformation_pipeline.py — Unit tests for TransformationPipeline (Story 4.4).
"""

import numpy as np
import pandas as pd
import pytest

from kiteml.pipeline import TransformationPipeline


def test_transformation_pipeline_fit_transform():
    df = pd.DataFrame(
        {
            "id": list(range(20)),
            "constant_col": [10] * 20,
            "price": [10.0, 20.0, np.nan, 40.0, 50.0] * 4,
            "quantity": [1, 2, 3, 4, 5] * 4,
            "category": ["A", "B", "A", "B", "A"] * 4,
            "date": pd.date_range("2025-01-01", periods=20, freq="D"),
            "target": [0, 1] * 10,
        }
    )

    pipeline = TransformationPipeline()
    transformed = pipeline.fit_transform(df, target="target", problem_type="classification")

    assert isinstance(transformed, pd.DataFrame)
    assert not transformed.empty
    assert "constant_col" not in transformed.columns
    assert "id" not in transformed.columns
    assert transformed.isna().sum().sum() == 0
