"""
test_rule_selector.py — Unit tests for RuleSelector (Story 4.3).
"""

import pandas as pd
import pytest

from kiteml.feature_selection import FSRuleEngine, RuleSelector, SelectionDecision


def test_rule_selector_constant_feature():
    df = pd.DataFrame({"const_col": [10] * 20})
    selector = RuleSelector()
    rules = FSRuleEngine()

    dec, score, reason = selector.evaluate("const_col", df, None, rules)

    assert dec == SelectionDecision.REMOVE
    assert score == 0.0
    assert "Constant feature" in reason


def test_rule_selector_identifier_feature():
    df = pd.DataFrame({"customer_id": [f"CUST_{i}" for i in range(100)]})
    selector = RuleSelector()
    rules = FSRuleEngine()

    dec, score, reason = selector.evaluate("customer_id", df, None, rules)

    assert dec == SelectionDecision.REMOVE
    assert "Identifier column" in reason
