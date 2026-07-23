"""Unit tests for ValidationResult."""

import json

from kiteml.validation.severity import ValidationSeverity
from kiteml.validation.validation_result import ValidationResult


def test_validation_result_passed_and_errors():
    res = ValidationResult(validator_name="TestValidator")
    assert res.passed is True
    assert res.has_errors() is False

    res.add_warning(title="Minor Warning", description="Low missing value count")
    assert res.passed is True
    assert res.has_warnings() is True

    res.add_error(title="Critical Error", description="Dataset is empty")
    assert res.passed is False
    assert res.has_errors() is True
    assert len(res.errors) == 1
    assert len(res.warnings) == 1


def test_validation_result_summary():
    res = ValidationResult(validator_name="SummaryTest")
    res.add_info("Info Title", "Info Desc")
    res.add_warning("Warn Title", "Warn Desc")
    res.add_error("Error Title", "Error Desc")
    res.add_critical("Crit Title", "Crit Desc")

    summary = res.summary
    assert summary["info"] == 1
    assert summary["warning"] == 1
    assert summary["error"] == 1
    assert summary["critical"] == 1


def test_validation_result_merge():
    res1 = ValidationResult(validator_name="Val1", execution_time=0.1)
    res1.add_warning("W1", "Warn 1")

    res2 = ValidationResult(validator_name="Val2", execution_time=0.2)
    res2.add_error("E1", "Error 1")

    merged = res1.merge(res2)
    assert merged.validator_name == "Val1+Val2"
    assert merged.execution_time == 0.3
    assert len(merged.messages) == 2
    assert merged.passed is False


def test_validation_result_json_serialization():
    res = ValidationResult(validator_name="JsonTest")
    res.add_info("Test Info", "Detail info text")

    d = res.to_dict()
    assert d["validator_name"] == "JsonTest"
    assert d["passed"] is True

    json_str = res.to_json()
    parsed = json.loads(json_str)
    assert parsed["validator_name"] == "JsonTest"
