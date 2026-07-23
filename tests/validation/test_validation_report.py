"""Unit tests for ValidationReport."""

import json

from kiteml.validation.validation_report import ValidationReport
from kiteml.validation.validation_result import ValidationResult


def test_validation_report_aggregation():
    report = ValidationReport(dataset_metadata={"n_rows": 100, "n_cols": 5, "memory_mb": 1.2})

    res1 = ValidationResult(validator_name="DatasetValidator")
    res1.add_info("Dataset Loaded", "Dataset successfully loaded.")
    report.add_result(res1)

    res2 = ValidationResult(validator_name="QualityValidator")
    res2.add_warning("Missing Values", "2 columns have missing values.")
    report.add_result(res2)

    assert report.passed is True
    assert report.has_errors() is False
    assert report.has_warnings() is True
    assert report.summary["warnings"] == 1

    terminal_text = report.render_terminal()
    assert "KiteML Dataset Validation" in terminal_text
    assert "Rows: 100" in terminal_text
    assert "Columns: 5" in terminal_text
    assert "Memory: 1.2 MB" in terminal_text
    assert "Ready for Training" in terminal_text


def test_validation_report_error_terminal():
    report = ValidationReport(dataset_metadata={"n_rows": 0, "n_cols": 0, "memory_mb": 0.0})

    res = ValidationResult(validator_name="DatasetValidator")
    res.add_error("Empty Dataset", "Dataset is completely empty.")
    report.add_result(res)

    assert report.passed is False
    assert report.has_errors() is True

    terminal_text = report.render_terminal()
    assert "Validation Failed" in terminal_text
    assert "❌ Empty Dataset" in terminal_text


def test_validation_report_to_dict_and_json():
    report = ValidationReport(dataset_metadata={"n_rows": 50})
    res = ValidationResult(validator_name="TestVal")
    res.add_info("Check Passed", "All clear")
    report.add_result(res)

    d = report.to_dict()
    assert d["passed"] is True
    assert len(d["validators"]) == 1

    json_str = report.to_json()
    parsed = json.loads(json_str)
    assert parsed["passed"] is True
