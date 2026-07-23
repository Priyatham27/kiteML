"""
test_target_validator.py — Comprehensive unit tests for TargetValidator (Story 2.3).
"""

import numpy as np
import pandas as pd
import pytest

from kiteml.validation.target_validator import TargetValidator


@pytest.fixture
def validator():
    return TargetValidator()


# ── Happy Path Tests ─────────────────────────────────────────────────────────


def test_binary_classification_target(validator):
    df = pd.DataFrame({"age": [20, 30, 40, 50, 60], "churn": [0, 1, 0, 1, 0]})
    res = validator.validate(df, target="churn", problem_type="classification")

    assert res.passed is True
    assert res.has_errors() is False
    intel = res.statistics["target_intelligence"]
    assert intel["target_name"] == "churn"
    assert intel["problem_type"] == "classification"
    assert intel["n_unique"] == 2
    assert intel["missing_pct"] == 0.0
    assert intel["recommended_metric"] == "Accuracy"
    assert intel["recommended_stratified_split"] is True
    assert res.statistics["health_score"] == 100


def test_multiclass_classification_target(validator):
    df = pd.DataFrame({"target": ["cat", "dog", "bird", "cat", "dog", "bird"]})
    res = validator.validate(df, target="target", problem_type="classification")

    assert res.passed is True
    intel = res.statistics["target_intelligence"]
    assert intel["n_unique"] == 3
    assert intel["problem_type"] == "classification"


def test_numeric_regression_target(validator):
    df = pd.DataFrame({"price": [100.0, 250.0, 400.0, 150.0, 320.0]})
    res = validator.validate(df, target="price", problem_type="regression")

    assert res.passed is True
    intel = res.statistics["target_intelligence"]
    assert intel["problem_type"] == "regression"
    assert intel["min"] == 100.0
    assert intel["max"] == 400.0
    assert intel["recommended_metric"] == "RMSE"
    assert intel["recommended_stratified_split"] is False


def test_auto_infer_problem_type(validator):
    # Continuous numeric -> auto infer regression
    df_reg = pd.DataFrame({"value": np.linspace(1, 100, 50)})
    res_reg = validator.validate(df_reg, target="value")
    assert res_reg.statistics["target_intelligence"]["problem_type"] == "regression"

    # Categorical string -> auto infer classification
    df_cls = pd.DataFrame({"label": ["A", "B", "A", "B"]})
    res_cls = validator.validate(df_cls, target="label")
    assert res_cls.statistics["target_intelligence"]["problem_type"] == "classification"


# ── Failure Case Tests ───────────────────────────────────────────────────────


def test_target_not_specified(validator):
    df = pd.DataFrame({"a": [1, 2, 3]})
    res = validator.validate(df, target=None)

    assert res.passed is False
    assert any(msg.rule_id == "KML-T001" for msg in res.messages)
    assert res.statistics["health_score"] == 0


def test_target_not_found(validator):
    df = pd.DataFrame({"a": [1, 2, 3]})
    res = validator.validate(df, target="non_existent")

    assert res.passed is False
    assert any(msg.rule_id == "KML-T002" for msg in res.messages)
    assert res.statistics["health_score"] == 0


def test_target_completely_empty(validator):
    df = pd.DataFrame({"target": [np.nan, np.nan, np.nan]})
    res = validator.validate(df, target="target")

    assert res.passed is False
    assert any(msg.rule_id == "KML-T003" for msg in res.messages)
    assert res.statistics["health_score"] == 0


def test_target_excessive_missing_values(validator):
    # >20% missing -> ERROR
    df = pd.DataFrame({"target": [1, 2, np.nan, np.nan, np.nan]})  # 60% missing
    res = validator.validate(df, target="target")

    assert res.passed is False
    assert any(msg.rule_id == "KML-T004" and msg.severity == "error" for msg in res.messages)


def test_single_class_classification(validator):
    df = pd.DataFrame({"label": [1, 1, 1, 1]})
    res = validator.validate(df, target="label", problem_type="classification")

    assert res.passed is False
    assert any(msg.rule_id == "KML-T005" for msg in res.messages)
    assert res.statistics["health_score"] == 0


def test_constant_regression_target(validator):
    df = pd.DataFrame({"price": [100.0, 100.0, 100.0]})
    res = validator.validate(df, target="price", problem_type="regression")

    assert res.passed is False
    assert any(msg.rule_id == "KML-T007" for msg in res.messages)
    assert res.statistics["health_score"] == 0


def test_non_numeric_regression_target(validator):
    df = pd.DataFrame({"price": ["$100", "$200", "$300"]})
    res = validator.validate(df, target="price", problem_type="regression")

    assert res.passed is False
    assert any(msg.rule_id == "KML-T008" for msg in res.messages)
    assert res.statistics["health_score"] == 0


# ── Warning Case Tests ───────────────────────────────────────────────────────


def test_class_imbalance_warning(validator):
    # 1 positive out of 50 samples (2%) -> Severe imbalance
    labels = [0] * 49 + [1]
    df = pd.DataFrame({"target": labels})
    res = validator.validate(df, target="target", problem_type="classification")

    assert res.passed is True
    assert res.has_warnings() is True
    assert any(msg.rule_id == "KML-T006" for msg in res.messages)
    intel = res.statistics["target_intelligence"]
    assert intel["recommended_metric"] == "F1 Score"


def test_excessive_classes_warning(validator):
    # >100 classes for classification
    df = pd.DataFrame({"target": [f"class_{i}" for i in range(150)]})
    res = validator.validate(df, target="target", problem_type="classification")

    assert res.passed is True
    assert res.has_warnings() is True
    assert any(msg.rule_id == "KML-T009" for msg in res.messages)


def test_identifier_like_target(validator):
    df = pd.DataFrame({"customer_id": [f"CUST_{i}" for i in range(50)]})
    res = validator.validate(df, target="customer_id", problem_type="classification")

    assert res.passed is True
    assert res.has_warnings() is True
    assert any(msg.rule_id == "KML-T010" for msg in res.messages)


def test_moderate_missing_percentage(validator):
    # 10% missing (1/10) -> WARNING
    df = pd.DataFrame({"target": [1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan]})
    res = validator.validate(df, target="target")

    assert res.passed is True
    assert res.has_warnings() is True
    assert any(msg.rule_id == "KML-T004" for msg in res.messages)
