"""Unit tests for ValidationEngine."""

import pandas as pd

from kiteml.validation.engine import ValidationEngine
from kiteml.validation.message import ValidationMessage
from kiteml.validation.rule import ValidationRule
from kiteml.validation.severity import ValidationSeverity
from kiteml.validation.validator import BaseValidator


class EngineRule1(ValidationRule):
    rule_id = "ENG01"
    name = "Engine Rule 1"

    def check(self, df: pd.DataFrame, **kwargs):
        if len(df) == 0:
            return ValidationMessage(
                severity=ValidationSeverity.ERROR,
                title="No Rows",
                description="DataFrame has no rows.",
            )
        return None


class EngineRule2(ValidationRule):
    rule_id = "ENG02"
    name = "Engine Rule 2"

    def check(self, df: pd.DataFrame, **kwargs):
        if len(df.columns) < 2:
            return ValidationMessage(
                severity=ValidationSeverity.WARNING,
                title="Single Column",
                description="DataFrame has less than 2 columns.",
            )
        return None


class EngineValidator1(BaseValidator):
    name = "Validator1"

    def __init__(self):
        super().__init__(rules=[EngineRule1()])


class EngineValidator2(BaseValidator):
    name = "Validator2"

    def __init__(self):
        super().__init__(rules=[EngineRule2()])


def test_validation_engine_pipeline():
    engine = ValidationEngine()
    engine.add_validator(EngineValidator1())
    engine.add_validator(EngineValidator2())

    assert len(engine) == 2

    df_valid = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    report = engine.validate(df_valid)

    assert report.passed is True
    assert len(report.results) == 2
    assert report.dataset_metadata["n_rows"] == 2
    assert report.dataset_metadata["n_cols"] == 2

    df_single_col = pd.DataFrame({"a": [1, 2]})
    report_single = engine.validate(df_single_col)
    assert report_single.passed is True
    assert report_single.has_warnings() is True

    df_empty = pd.DataFrame()
    report_empty = engine.validate(df_empty)
    assert report_empty.passed is False
    assert report_empty.has_errors() is True


def test_validation_engine_stop_on_error():
    engine = ValidationEngine([EngineValidator1(), EngineValidator2()])
    df_empty = pd.DataFrame()

    report = engine.validate(df_empty, stop_on_error=True)
    # Since EngineValidator1 fails on empty DataFrame, EngineValidator2 should not execute
    assert len(report.results) == 1
