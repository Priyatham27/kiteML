"""
test_integration.py — Integration tests for error & result suggestions API (Story 3.5).
"""

import pandas as pd
import pytest

import kiteml
from kiteml.exceptions import TargetError


def test_exception_suggestions_api():
    err = TargetError(
        "Target column 'prcie' not found",
        context={"target": "prcie", "available_columns": ["Price", "Age", "Salary"]},
    )

    suggs = err.suggestions()
    assert len(suggs) > 0
    assert "Price" in suggs[0].title
    assert len(suggs[0].why) > 0


def test_result_suggestions_api():
    res = kiteml.Result(
        model=None,
        metrics={},
        report="",
        problem_type="classification",
        feature_names=["age", "const_feature"],
    )

    suggs = res.suggestions()
    assert isinstance(suggs, list)
