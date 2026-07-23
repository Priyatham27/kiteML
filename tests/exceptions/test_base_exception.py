"""
test_base_exception.py — Unit tests for KiteML Exception Framework (Story 3.1).
"""

import json

import pandas as pd
import pytest

from kiteml.exceptions import (
    CLIError,
    ConfigurationError,
    DatasetError,
    DeploymentError,
    ErrorContext,
    KiteMLError,
    PredictionError,
    PreprocessingError,
    SchemaError,
    TargetError,
    TrainingError,
    ValidationError,
    build_dataframe_context,
    wrap_exception,
)
from kiteml.exceptions.codes import (
    KML_ERR_DATASET,
    KML_ERR_GENERIC,
    KML_ERR_TARGET,
)


def test_base_exception_initialization():
    err = KiteMLError(
        message="Test error message",
        error_code="KML-E001",
        severity="error",
        suggestion="Fix the problem",
        details="Extra details",
        help_url="https://kiteml.org/docs/errors",
    )

    assert isinstance(err, Exception)
    assert err.message == "Test error message"
    assert err.error_code == "KML-E001"
    assert err.severity == "error"
    assert err.suggestion == "Fix the problem"
    assert err.details == "Extra details"
    assert err.help_url == "https://kiteml.org/docs/errors"
    assert repr(err) == "<KiteMLError [KML-E001]: Test error message>"
    assert "[KML-E001] Test error message" in str(err)
    assert "Suggestion: Fix the problem" in str(err)


def test_error_context_model():
    ctx = ErrorContext(
        operation="training",
        dataset_name="churn.csv",
        target="churn",
        available_columns=["age", "salary", "churn"],
        row_count=100,
        column_count=3,
        feature_name="age",
        model_name="RandomForest",
        metadata={"extra_info": "val"},
    )

    d = ctx.to_dict()
    assert d["operation"] == "training"
    assert d["available_columns"] == ["age", "salary", "churn"]
    assert d["metadata"] == {"extra_info": "val"}

    reconstructed = ErrorContext.from_dict(d)
    assert reconstructed.dataset_name == "churn.csv"
    assert reconstructed.target == "churn"


def test_context_coercion_in_kiteml_error():
    # Pass raw dict -> automatically coerced to ErrorContext
    raw_dict = {
        "target": "price",
        "available_columns": ["age", "salary"],
        "custom_key": "custom_val",
    }
    err = KiteMLError(message="Target missing", context=raw_dict)

    assert isinstance(err.context, ErrorContext)
    assert err.context.target == "price"
    assert err.context.available_columns == ["age", "salary"]
    assert err.context.metadata.get("custom_key") == "custom_val"


def test_serialization_to_dict_and_json():
    err = KiteMLError(
        message="Target column missing",
        error_code=KML_ERR_TARGET,
        suggestion="Use an existing column",
        context={"target": "price", "available_columns": ["age", "salary"]},
    )

    d = err.to_dict()
    assert d["error_class"] == "KiteMLError"
    assert d["message"] == "Target column missing"
    assert d["error_code"] == KML_ERR_TARGET
    assert d["suggestion"] == "Use an existing column"
    assert d["context"]["target"] == "price"

    json_str = err.to_json(indent=2)
    parsed = json.loads(json_str)
    assert parsed["error_code"] == KML_ERR_TARGET
    assert parsed["context"]["target"] == "price"


def test_domain_exceptions_hierarchy():
    domain_exceptions = [
        (DatasetError("ds err"), KML_ERR_DATASET),
        (TargetError("target err"), KML_ERR_TARGET),
        (SchemaError("schema err"), "KML-S000"),
        (ValidationError("val err"), "KML-V000"),
        (PreprocessingError("prep err"), "KML-P000"),
        (TrainingError("train err"), "KML-M000"),
        (DeploymentError("deploy err"), "KML-DP000"),
        (PredictionError("pred err"), "KML-I000"),
        (CLIError("cli err"), "KML-C000"),
        (ConfigurationError("config err"), "KML-CFG000"),
    ]

    for err_instance, expected_code in domain_exceptions:
        assert isinstance(err_instance, KiteMLError)
        assert isinstance(err_instance, Exception)
        assert err_instance.error_code == expected_code


def test_exception_chaining():
    cause = ValueError("Original pandas error")
    try:
        try:
            raise cause
        except ValueError as exc:
            raise TargetError(
                message="Target column failed validation",
                suggestion="Specify a valid target",
            ) from exc
    except TargetError as err:
        assert err.__cause__ is cause
        d = err.to_dict()
        assert "cause" in d
        assert "ValueError: Original pandas error" in d["cause"]


def test_utils_helpers():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    ctx = build_dataframe_context(df, target="b", operation="validate")

    assert ctx.row_count == 2
    assert ctx.column_count == 2
    assert ctx.available_columns == ["a", "b"]
    assert ctx.target == "b"
    assert ctx.operation == "validate"

    orig_exc = KeyError("missing_col")
    wrapped = wrap_exception(orig_exc, error_class=DatasetError, message="Dataset column missing")

    assert isinstance(wrapped, DatasetError)
    assert wrapped.message == "Dataset column missing"
    assert wrapped.__cause__ is orig_exc
