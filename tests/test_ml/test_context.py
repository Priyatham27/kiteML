"""
test_context.py — Unit tests for MLContext (Story 5.8).
"""

import pandas as pd
import pytest

from kiteml.ml import MLContext


def test_ml_context():
    df = pd.DataFrame({"a": [1], "b": [2]})
    ctx = MLContext(dataframe=df, target_column="b")
    assert ctx.target_column == "b"
    assert ctx.problem_type == "classification"
