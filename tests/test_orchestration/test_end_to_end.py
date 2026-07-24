"""
test_end_to_end.py — End-to-end integration tests for KiteMLPipeline (Story 4.7 & Epic 4 Completion).
"""

import numpy as np
import pandas as pd
import pytest

from kiteml import KiteMLPipeline


def test_full_automl_pipeline_end_to_end():
    df = pd.DataFrame(
        {
            "id": list(range(20)),
            "constant_col": [99] * 20,
            "price": [10.0, 20.0, np.nan, 40.0, 50.0] * 4,
            "quantity": [1, 2, 3, 4, 5] * 4,
            "category": ["A", "B", "A", "B", "A"] * 4,
            "date": pd.date_range("2025-01-01", periods=20, freq="D"),
            "target": [0, 1] * 10,
        }
    )

    pipeline = KiteMLPipeline()
    result = pipeline.build(dataframe=df, target="target", problem_type="classification")

    assert result.transformed_df is not None
    assert "constant_col" not in result.transformed_df.columns
    assert "id" not in result.transformed_df.columns
    assert result.transformed_df.isna().sum().sum() == 0
    assert result.report is not None
