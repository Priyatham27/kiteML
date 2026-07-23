"""
test_schema_validator.py — Comprehensive unit tests for SchemaValidator (Story 2.4).
"""

import numpy as np
import pandas as pd
import pytest

from kiteml.validation.schema_validator import SchemaValidator


@pytest.fixture
def validator():
    return SchemaValidator()


# ── Happy Path Tests ─────────────────────────────────────────────────────────


def test_clean_schema_validation(validator):
    df = pd.DataFrame(
        {
            "age": [25, 30, 35, 40],
            "income": [50000.0, 60000.0, 75000.0, 90000.0],
            "gender": ["M", "F", "F", "M"],
            "target": [0, 1, 0, 1],
        }
    )
    res = validator.validate(df, target="target")

    assert res.passed is True
    assert res.has_errors() is False
    assert res.statistics["n_features"] == 3
    profiles = res.statistics["feature_profiles"]
    assert "target" not in profiles  # Target column excluded
    assert profiles["age"]["semantic_type"] == "numeric"
    assert profiles["age"]["recommendation"] == "Standardize"
    assert profiles["gender"]["semantic_type"] == "categorical"
    assert profiles["gender"]["recommendation"] == "One-Hot Encode"


def test_mixed_types_dataset_schema(validator):
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0],
            "is_active": [True, False, True, False],
            "dt": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
            "review": [
                "This product is amazing and works very well for everyone.",
                "Terrible experience, broke after two days of usage.",
                "Decent performance for the price point, satisfied overall.",
                "High quality build and excellent customer support team.",
            ],
            "label": [1, 0, 1, 1],
        }
    )
    res = validator.validate(df, target="label")

    assert res.passed is True
    profiles = res.statistics["feature_profiles"]
    assert profiles["is_active"]["semantic_type"] == "boolean"
    assert profiles["is_active"]["recommendation"] == "Encode Binary (0/1)"
    assert profiles["dt"]["semantic_type"] == "datetime"
    assert profiles["dt"]["recommendation"] == "Extract Year/Month/Day"
    assert profiles["review"]["semantic_type"] == "text"
    assert profiles["review"]["recommendation"] == "TF-IDF Vectorization"


# ── Failure Case Tests ───────────────────────────────────────────────────────


def test_empty_feature_name(validator):
    df = pd.DataFrame([[1, 2]], columns=["col_a", "Unnamed: 1"])
    res = validator.validate(df)

    assert res.passed is False
    assert any(msg.rule_id == "KML-S001" for msg in res.messages)


def test_duplicate_feature_names(validator):
    df = pd.DataFrame([[1, 2]], columns=["feature_a", "feature_a"])
    res = validator.validate(df)

    assert res.passed is False
    assert any(msg.rule_id == "KML-S002" for msg in res.messages)


def test_unsupported_datatype(validator):
    df = pd.DataFrame({"a": [1, 2], "b": [{"dict_key": "val"}, [1, 2]]})
    res = validator.validate(df)

    assert res.passed is False
    assert any(msg.rule_id == "KML-S003" for msg in res.messages)


def test_infinite_numeric_feature(validator):
    df = pd.DataFrame({"feature": [1.0, np.inf, 3.0, -np.inf]})
    res = validator.validate(df)

    assert res.passed is False
    assert any(msg.rule_id == "KML-S011" for msg in res.messages)


# ── Warning Case Tests ───────────────────────────────────────────────────────


def test_constant_feature_warning(validator):
    df = pd.DataFrame({"constant_col": ["India", "India", "India", "India"]})
    res = validator.validate(df)

    assert res.passed is True
    assert res.has_warnings() is True
    assert any(msg.rule_id == "KML-S004" for msg in res.messages)
    profiles = res.statistics["feature_profiles"]
    assert profiles["constant_col"]["is_constant"] is True
    assert profiles["constant_col"]["recommendation"] == "Remove (Constant)"


def test_high_cardinality_feature_warning(validator):
    # >50% unique ratio in categorical feature
    df = pd.DataFrame({"high_card": [f"val_{i}" for i in range(30)]})
    res = validator.validate(df)

    assert res.passed is True
    assert res.has_warnings() is True
    assert any(msg.rule_id == "KML-S005" for msg in res.messages)
    profiles = res.statistics["feature_profiles"]
    assert profiles["high_card"]["cardinality"] in ("high", "identifier")


def test_identifier_feature_warning(validator):
    df = pd.DataFrame({"customer_id": [f"CUST_{i:04d}" for i in range(50)]})
    res = validator.validate(df)

    assert res.passed is True
    assert res.has_warnings() is True
    assert any(msg.rule_id == "KML-S006" for msg in res.messages)
    profiles = res.statistics["feature_profiles"]
    assert profiles["customer_id"]["is_identifier"] is True
    assert profiles["customer_id"]["recommendation"] == "Remove (Identifier)"


def test_mixed_datatype_feature_warning(validator):
    df = pd.DataFrame({"mixed": [10, "twenty", 30, "forty"]})
    res = validator.validate(df)

    assert res.passed is True
    assert res.has_warnings() is True
    assert any(msg.rule_id == "KML-S010" for msg in res.messages)


def test_extremely_sparse_feature_warning(validator):
    # >70% missing
    df = pd.DataFrame({"sparse": [1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]})
    res = validator.validate(df)

    assert res.passed is True
    assert res.has_warnings() is True
    assert any(msg.rule_id == "KML-S012" for msg in res.messages)


# ── Info Case Tests ──────────────────────────────────────────────────────────


def test_datetime_feature_info(validator):
    df = pd.DataFrame({"signup_date": pd.to_datetime(["2026-01-01", "2026-01-02"])})
    res = validator.validate(df)

    assert res.passed is True
    assert any(msg.rule_id == "KML-S007" for msg in res.messages)


def test_boolean_feature_info(validator):
    df = pd.DataFrame({"flag": [True, False, True, False]})
    res = validator.validate(df)

    assert res.passed is True
    assert any(msg.rule_id == "KML-S008" for msg in res.messages)


def test_text_feature_info(validator):
    df = pd.DataFrame(
        {
            "description": [
                "Detailed textual product review detailing features and quality.",
                "Comprehensive user feedback explaining performance and value.",
            ]
        }
    )
    res = validator.validate(df)

    assert res.passed is True
    assert any(msg.rule_id == "KML-S009" for msg in res.messages)
