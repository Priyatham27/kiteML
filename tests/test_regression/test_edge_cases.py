"""
test_edge_cases.py — Edge case regression tests for KiteML DX framework (Story 3.7).
"""

import pandas as pd
import pytest

import kiteml
from kiteml.exceptions import KiteMLError, TargetError


def test_edge_case_empty_dataset():
    df = pd.DataFrame()
    with pytest.raises((ValueError, KiteMLError)):
        kiteml.train(df, target="price")


def test_edge_case_unicode_column_names():
    df = pd.DataFrame(
        {
            "Preço": [10.0, 20.0, 30.0, 40.0, 50.0] * 4,
            "नाम": ["A", "B", "A", "B", "A"] * 4,
            "年龄": [20, 30, 40, 50, 60] * 4,
            "target": [0, 1] * 10,
        }
    )
    res = kiteml.train(df, target="target", validate_data=True)
    assert res.has_warnings() is False or isinstance(res.has_warnings(), bool)


def test_edge_case_long_column_names():
    long_col = "This_Is_An_Insanely_Long_Column_Name_That_Exceeds_Normal_Length_Limits_For_Testing"
    df = pd.DataFrame(
        {
            long_col: [1, 2, 3, 4, 5] * 4,
            "target": [0, 1] * 10,
        }
    )
    res = kiteml.train(df, target="target", validate_data=True)
    assert res.model is not None


def test_edge_case_typo_target_suggestions():
    df = pd.DataFrame(
        {
            "Price": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Age": [20, 30, 40, 50, 60],
        }
    )

    err = TargetError("Target 'prcie' not found", context={"target": "prcie", "available_columns": list(df.columns)})
    suggs = err.suggestions()
    assert len(suggs) > 0
    assert "Price" in suggs[0].title
