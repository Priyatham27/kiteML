"""
test_dataset_validator.py — Comprehensive unit tests for DatasetValidator (Story 2.2).
"""

import numpy as np
import pandas as pd
import pytest

from kiteml.validation.dataset_validator import DatasetValidator


@pytest.fixture
def validator():
    return DatasetValidator()


# ── Happy Path Tests ─────────────────────────────────────────────────────────


def test_clean_dataset(validator):
    df = pd.DataFrame(
        {
            "age": [25, 30, 35, 40],
            "income": [50000.0, 60000.0, 75000.0, 90000.0],
            "city": ["NY", "LA", "SF", "CHI"],
        }
    )
    res = validator.validate(df)

    assert res.passed is True
    assert res.has_errors() is False
    assert res.has_warnings() is False
    assert res.statistics["n_rows"] == 4
    assert res.statistics["n_cols"] == 3
    assert res.statistics["health_score"] == 100
    assert "Excellent" in res.statistics["health_rating"]


def test_large_dataset(validator):
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "feature_a": np.random.randn(10000),
            "feature_b": np.random.randint(0, 100, size=10000),
        }
    )
    res = validator.validate(df)

    assert res.passed is True
    assert res.statistics["n_rows"] == 10000
    assert res.statistics["health_score"] == 100
    assert res.execution_time < 0.5  # Under 500ms requirement


# ── Failure Case Tests (Critical / Errors) ───────────────────────────────────


def test_none_dataset(validator):
    res = validator.validate(None)

    assert res.passed is False
    assert res.has_errors() is True
    assert any(msg.rule_id == "KML-D001" for msg in res.messages)
    assert res.statistics["health_score"] == 0


def test_non_dataframe_dataset(validator):
    res = validator.validate([1, 2, 3])

    assert res.passed is False
    assert any(msg.rule_id == "KML-D002" for msg in res.messages)
    assert res.statistics["health_score"] == 0


def test_empty_dataframe(validator):
    res = validator.validate(pd.DataFrame())

    assert res.passed is False
    assert any(msg.rule_id in ("KML-D003", "KML-D004", "KML-D005") for msg in res.messages)
    assert res.statistics["health_score"] == 0


def test_zero_rows_dataframe(validator):
    df = pd.DataFrame(columns=["a", "b"])
    res = validator.validate(df)

    assert res.passed is False
    assert any(msg.rule_id == "KML-D004" for msg in res.messages)


def test_zero_columns_dataframe(validator):
    df = pd.DataFrame(index=[0, 1, 2])
    res = validator.validate(df)

    assert res.passed is False
    assert any(msg.rule_id == "KML-D005" for msg in res.messages)


def test_duplicate_columns(validator):
    df = pd.DataFrame([[1, 2]], columns=["col_a", "col_a"])
    res = validator.validate(df)

    assert res.passed is False
    assert any(msg.rule_id == "KML-D006" for msg in res.messages)
    assert res.statistics["health_score"] == 90


def test_empty_column_names(validator):
    df = pd.DataFrame([[1, 2, 3]], columns=["col_a", "", "Unnamed: 2"])
    res = validator.validate(df)

    assert res.passed is False
    assert any(msg.rule_id == "KML-D007" for msg in res.messages)


def test_reserved_column_names(validator):
    df = pd.DataFrame([[1, 2]], columns=["feature1", "_kiteml_"])
    res = validator.validate(df)

    assert res.passed is True  # Reserved names cause WARNING, not blocking ERROR
    assert res.has_warnings() is True
    assert any(msg.rule_id == "KML-D008" for msg in res.messages)
    assert res.statistics["health_score"] == 99


def test_duplicate_rows(validator):
    df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 10, 20]})
    res = validator.validate(df)

    assert res.passed is True
    assert res.has_warnings() is True
    assert any(msg.rule_id == "KML-D009" for msg in res.messages)
    assert res.statistics["duplicate_rows"] == 1
    assert res.statistics["health_score"] == 98


def test_empty_rows(validator):
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [10.0, np.nan, 30.0]})
    res = validator.validate(df)

    assert res.passed is True
    assert res.has_warnings() is True
    assert any(msg.rule_id == "KML-D010" for msg in res.messages)
    assert res.statistics["empty_rows"] == 1
    assert res.statistics["health_score"] == 95


def test_infinite_values(validator):
    df = pd.DataFrame({"a": [1.0, np.inf, -np.inf, 4.0], "b": [10, 20, 30, 40]})
    res = validator.validate(df)

    assert res.passed is False
    assert any(msg.rule_id == "KML-D011" for msg in res.messages)
    assert res.statistics["health_score"] == 80


def test_unsupported_cell_objects(validator):
    df = pd.DataFrame({"a": [1, 2], "b": [{"key": "val"}, [1, 2]]})
    res = validator.validate(df)

    assert res.passed is False
    assert any(msg.rule_id == "KML-D012" for msg in res.messages)
    assert res.statistics["health_score"] == 90


# ── Edge Case Tests ──────────────────────────────────────────────────────────


def test_single_row_single_column(validator):
    df = pd.DataFrame({"single": [42]})
    res = validator.validate(df)

    assert res.passed is True
    assert res.statistics["n_rows"] == 1
    assert res.statistics["n_cols"] == 1


def test_all_nan_dataset(validator):
    df = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
    res = validator.validate(df)

    assert res.passed is True
    assert res.has_warnings() is True
    assert any(msg.rule_id == "KML-D010" for msg in res.messages)
    assert res.statistics["missing_cells"] == 4
