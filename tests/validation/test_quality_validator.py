"""
test_quality_validator.py — Comprehensive unit tests for QualityValidator (Story 2.5).
"""

import numpy as np
import pandas as pd
import pytest

from kiteml.validation.quality_validator import QualityValidator


@pytest.fixture
def validator():
    return QualityValidator()


# ── Happy Path Tests ─────────────────────────────────────────────────────────


def test_clean_high_quality_dataset(validator):
    df = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50, 55, 60],
            "income": [45000.0, 52000.0, 61000.0, 70000.0, 81000.0, 92000.0, 105000.0, 120000.0],
            "churn": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    res = validator.validate(df, target="churn", problem_type="classification")

    assert res.passed is True
    assert res.has_errors() is False
    assert res.statistics["health_score"] >= 95
    assert res.statistics["health_grade"] == "A+"
    profile = res.statistics["quality_profile"]
    assert profile["overall_score"] >= 95
    assert profile["overall_grade"] == "A+"
    assert profile["missing_summary"]["missing_cells"] == 0
    assert profile["duplicate_summary"]["duplicate_rows"] == 0


def test_clean_regression_quality_dataset(validator):
    df = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "x2": [10.0, 20.0, 15.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0],
            "target": [2.5, 4.8, 6.9, 9.1, 11.2, 13.5, 15.8, 18.0, 20.1, 22.4],
        }
    )
    res = validator.validate(df, target="target", problem_type="regression")

    assert res.passed is True
    assert res.statistics["health_grade"] in ("A+", "A")


# ── Warning Case Tests ───────────────────────────────────────────────────────


def test_moderate_missing_values_warning(validator):
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "b": [np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, np.nan],
        }
    )
    res = validator.validate(df)

    assert res.passed is True
    assert res.has_warnings() is True
    assert any(msg.rule_id == "KML-Q001" for msg in res.messages)


def test_empty_rows_warning(validator):
    # 1 empty row in 10 rows -> 10% missing cells overall (WARNING, passed=True)
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, np.nan],
            "b": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, np.nan],
        }
    )
    res = validator.validate(df)

    assert res.passed is True
    assert res.has_warnings() is True
    assert any(msg.rule_id == "KML-Q003" for msg in res.messages)
    assert res.statistics["quality_profile"]["missing_summary"]["missing_cells"] > 0


def test_statistical_outliers_warning(validator):
    # Baseline normal data with 2 extreme outliers in 20 samples (>5%)
    data = list(np.linspace(10, 20, 18)) + [500.0, 1000.0]
    df = pd.DataFrame({"num": data})
    res = validator.validate(df)

    assert res.passed is True
    assert res.has_warnings() is True
    assert any(msg.rule_id == "KML-Q004" for msg in res.messages)


def test_high_correlation_warning(validator):
    x = np.linspace(1, 100, 20)
    y = x * 2.0 + 0.001 * np.random.randn(20)  # r > 0.99
    df = pd.DataFrame({"feat_1": x, "feat_2": y})
    res = validator.validate(df)

    assert res.passed is True
    assert res.has_warnings() is True
    assert any(msg.rule_id == "KML-Q005" for msg in res.messages)
    corr_summary = res.statistics["quality_profile"]["correlation_summary"]
    assert len(corr_summary["high_correlation_pairs"]) > 0


def test_near_zero_variance_warning(validator):
    # 99.5% single value (199 out of 200)
    data = [1.0] * 199 + [2.0]
    df = pd.DataFrame({"feat": data})
    res = validator.validate(df)

    assert res.passed is True
    assert res.has_warnings() is True
    assert any(msg.rule_id == "KML-Q006" for msg in res.messages)


def test_duplicate_rows_warning(validator):
    df = pd.DataFrame(
        {
            "a": [1, 1, 1, 2],
            "b": [10, 10, 10, 20],
        }
    )
    res = validator.validate(df)

    assert res.passed is True
    assert res.has_warnings() is True
    assert any(msg.rule_id == "KML-Q007" for msg in res.messages)
    assert res.statistics["quality_profile"]["duplicate_summary"]["duplicate_rows"] == 2


def test_class_imbalance_warning(validator):
    df = pd.DataFrame({"target": [0] * 95 + [1] * 5})
    res = validator.validate(df, target="target", problem_type="classification")

    assert res.passed is True
    assert res.has_warnings() is True
    assert any(msg.rule_id == "KML-Q008" for msg in res.messages)


def test_data_consistency_pseudo_nulls_warning(validator):
    df = pd.DataFrame({"text_col": ["valid", "N/A", "null", "  ", "good"]})
    res = validator.validate(df)

    assert res.passed is True
    assert res.has_warnings() is True
    assert any(msg.rule_id == "KML-Q009" for msg in res.messages)


# ── Failure Case Tests ───────────────────────────────────────────────────────


def test_fully_empty_column_failure(validator):
    df = pd.DataFrame(
        {
            "valid": [1, 2, 3],
            "empty": [np.nan, np.nan, np.nan],
        }
    )
    res = validator.validate(df)

    assert res.passed is False
    assert any(msg.rule_id == "KML-Q002" for msg in res.messages)


# ── Health Score & Grade Tests ───────────────────────────────────────────────


def test_health_score_deductions_and_grades(validator):
    # Dataset with multiple quality issues -> Should deduct points and lower grade
    x = np.linspace(1, 100, 20)
    y = x * 2.0  # High correlation r=1.0 (-2)
    # Duplicate rows (-3)
    df = pd.DataFrame(
        {
            "a": list(x) + list(x[:5]),
            "b": list(y) + list(y[:5]),
            "c": [np.nan] * 8 + list(range(17)),  # Moderate missing values (-5)
        }
    )
    res = validator.validate(df)

    score = res.statistics["health_score"]
    grade = res.statistics["health_grade"]
    assert score < 95
    assert grade in ("A", "B", "C", "Needs Attention")
    assert len(res.statistics["recommendations"]) > 0
