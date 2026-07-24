"""
test_variance_selector.py — Unit tests for VarianceSelector (Story 4.3).
"""

import pandas as pd
import pytest

from kiteml.feature_selection import FSRuleEngine, SelectionDecision, VarianceSelector


def test_variance_selector():
    df = pd.DataFrame(
        {
            "zero_var": [1.0000000] * 20,
            "normal_var": list(range(20)),
        }
    )

    selector = VarianceSelector()
    rules = FSRuleEngine(min_variance=1e-4)

    dec1, score1, _ = selector.evaluate("zero_var", df, None, rules)
    assert dec1 == SelectionDecision.REMOVE

    dec2, score2, _ = selector.evaluate("normal_var", df, None, rules)
    assert dec2 == SelectionDecision.KEEP
    assert score2 == 90.0
