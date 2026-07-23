"""
test_integration.py — Integration tests for validation pipeline in train() and top-level validate() API (Story 2.6).
"""

import pandas as pd
import pytest

import kiteml
from kiteml import train
from kiteml.validation import validate


def test_standalone_validate_api():
    df = pd.DataFrame(
        {
            "feature_1": [10.0, 20.0, 30.0, 40.0, 50.0],
            "feature_2": ["X", "Y", "X", "Y", "X"],
            "target": [0, 1, 0, 1, 0],
        }
    )
    summary = validate(df, target="target", problem_type="classification")

    assert summary.passed is True
    assert summary.ready_for_training is True
    assert summary.health_score >= 90


def test_kiteml_top_level_validate_export():
    assert hasattr(kiteml, "validate")
    assert callable(kiteml.validate)


def test_train_automatic_validation_success():
    df = pd.DataFrame(
        {
            "x1": [float(i) for i in range(30)],
            "x2": [float(i * 10) for i in range(30)],
            "target": [0, 1] * 15,
        }
    )
    result = train(df, target="target", verbose=False)

    assert result is not None
    assert result.validation is not None
    assert result.validation.passed is True
    assert result.validation.ready_for_training is True


def test_train_automatic_validation_failure_raises():
    # Target column missing -> Validation fails -> train() raises ValueError
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(ValueError, match="Validation failed"):
        train(df, target="missing_target", verbose=False)
