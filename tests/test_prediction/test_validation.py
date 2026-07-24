"""
test_validation.py — Unit tests for SchemaAdapter (Story 5.6 Flagship Feature).
"""

import pandas as pd
import pytest

from kiteml.prediction import SchemaAdapter


def test_schema_adapter_column_reordering_and_extra_column_drop():
    adapter = SchemaAdapter()
    df = pd.DataFrame({"b": [2], "a": [1], "extra": [99]})
    adapted = adapter.adapt_schema(df, expected_columns=["a", "b"])

    assert list(adapted.columns) == ["a", "b"]
    assert adapted.iloc[0]["a"] == 1
    assert adapted.iloc[0]["b"] == 2
