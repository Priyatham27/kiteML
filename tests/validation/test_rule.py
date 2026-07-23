"""Unit tests for ValidationRule."""

import pandas as pd

from kiteml.validation.message import ValidationMessage
from kiteml.validation.rule import ValidationRule
from kiteml.validation.severity import ValidationSeverity


class SampleDummyRule(ValidationRule):
    rule_id = "R_DUMMY"
    name = "Dummy Rule"
    description = "Dummy rule for unit testing"

    def check(self, df: pd.DataFrame, **kwargs):
        if len(df) == 0:
            return ValidationMessage(
                severity=ValidationSeverity.ERROR,
                title="Empty Dataset",
                description="Dataset contains 0 rows.",
                rule_id=self.rule_id,
            )
        return None


def test_validation_rule_subclass():
    rule = SampleDummyRule()
    assert rule.rule_id == "R_DUMMY"
    assert rule.name == "Dummy Rule"

    df_empty = pd.DataFrame()
    msg = rule.check(df_empty)
    assert msg is not None
    assert msg.severity == ValidationSeverity.ERROR
    assert msg.title == "Empty Dataset"

    df_valid = pd.DataFrame({"a": [1, 2, 3]})
    assert rule.check(df_valid) is None
