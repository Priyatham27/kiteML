"""
test_rules.py — Unit tests for RuleEngine (Story 4.1).
"""

import pytest

from kiteml.preprocessing import RuleEngine


def test_rule_engine_defaults():
    rules = RuleEngine()
    assert rules.low_cardinality_threshold == 15
    assert rules.high_missing_drop_threshold == 0.80
    assert rules.high_skewness_threshold == 1.5


def test_rule_engine_customization():
    rules = RuleEngine(low_cardinality_threshold=10, high_missing_drop_threshold=0.50)
    assert rules.low_cardinality_threshold == 10
    assert rules.high_missing_drop_threshold == 0.50
