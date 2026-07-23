"""Unit tests for BaseValidator."""

import pandas as pd

from kiteml.validation.message import ValidationMessage
from kiteml.validation.rule import ValidationRule
from kiteml.validation.severity import ValidationSeverity
from kiteml.validation.validator import BaseValidator


class LocalDummyRule(ValidationRule):
    rule_id = "R_LOCAL"
    name = "Local Dummy Rule"

    def check(self, df: pd.DataFrame, **kwargs):
        if len(df) == 0:
            return ValidationMessage(
                severity=ValidationSeverity.ERROR,
                title="Empty Dataset",
                description="Dataset has 0 rows.",
            )
        return None


class SampleValidator(BaseValidator):
    name = "SampleValidator"
    description = "Test validator implementation"


def test_base_validator_execution():
    rule = LocalDummyRule()
    validator = SampleValidator(rules=[rule])

    df_valid = pd.DataFrame({"col": [10, 20]})
    res = validator.validate(df_valid)

    assert res.validator_name == "SampleValidator"
    assert res.passed is True
    assert res.statistics["n_rows"] == 2
    assert res.statistics["n_cols"] == 1
    assert res.execution_time >= 0.0

    df_empty = pd.DataFrame()
    res_empty = validator.validate(df_empty)
    assert res_empty.passed is False
    assert len(res_empty.errors) == 1
