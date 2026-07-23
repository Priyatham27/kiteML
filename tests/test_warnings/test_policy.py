"""
test_policy.py — Unit tests for WarningPolicy engine (Story 3.4).
"""

import pytest

from kiteml.exceptions import KiteMLError
from kiteml.warnings import DatasetWarning, SchemaWarning, WarningPolicy


def test_warning_policy_ignore():
    policy = WarningPolicy(code_actions={"KML-W-D001": "ignore"})
    w = DatasetWarning("High missing values", code="KML-W-D001")

    assert policy.process(w) is None


def test_warning_policy_error_escalation():
    policy = WarningPolicy(category_actions={"Schema": "error"})
    w = SchemaWarning("Constant feature detected", code="KML-W-S001")

    with pytest.raises(KiteMLError) as exc_info:
        policy.process(w)

    assert "Constant feature detected" in str(exc_info.value)
    assert exc_info.value.error_code == "KML-S001"
