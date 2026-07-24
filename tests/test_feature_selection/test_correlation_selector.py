"""
test_correlation_selector.py — Unit tests for CorrelationSelector (Story 4.3).
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from kiteml.feature_selection import CorrelationSelector, FSRuleEngine, SelectionDecision


def test_correlation_selector():
    df = pd.DataFrame({"colA": [1, 2, 3], "colB": [2, 4, 6]})

    mock_profile = MagicMock()
    mock_profile.correlations.matrix = pd.DataFrame(
        {
            "colA": [1.0, 0.98],
            "colB": [0.98, 1.0],
        },
        index=["colA", "colB"],
    )

    selector = CorrelationSelector()
    rules = FSRuleEngine(max_correlation=0.95)

    dec, score, reason = selector.evaluate("colA", df, mock_profile, rules)
    assert dec == SelectionDecision.REMOVE
    assert "High collinearity" in reason
