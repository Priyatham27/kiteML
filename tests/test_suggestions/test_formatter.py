"""
test_formatter.py — Unit tests for SuggestionFormatter (Story 3.5).
"""

import json

import pytest

from kiteml.suggestions.formatter import SuggestionFormatter
from kiteml.suggestions.result import Suggestion


def test_suggestion_formatter_modes():
    s = Suggestion(
        title="Use 'Price' as target",
        description="Matched target name",
        confidence=0.95,
        why=["Column name matched", "Exists in dataset"],
    )
    fmt = SuggestionFormatter()

    term_out = fmt.to_terminal([s])
    assert "KiteML Suggestions" in term_out
    assert "95% confidence" in term_out
    assert "Why?" in term_out
    assert "Column name matched" in term_out

    json_out = fmt.to_json([s])
    parsed = json.loads(json_out)
    assert len(parsed) == 1
    assert parsed[0]["confidence_percentage"] == 95
    assert len(parsed[0]["why"]) == 2
