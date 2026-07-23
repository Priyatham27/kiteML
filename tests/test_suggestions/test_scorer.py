"""
test_scorer.py — Unit tests for suggestion scoring and ranking (Story 3.5).
"""

import pytest

from kiteml.suggestions.result import Suggestion
from kiteml.suggestions.scorer import rank_suggestions


def test_rank_suggestions():
    s1 = Suggestion("Low confidence", "desc", 0.3)
    s2 = Suggestion("High confidence", "desc", 0.95)
    s3 = Suggestion("Medium confidence", "desc", 0.75)

    ranked = rank_suggestions([s1, s2, s3], min_confidence=0.5, top_k=2)

    assert len(ranked) == 2
    assert ranked[0].title == "High confidence"
    assert ranked[1].title == "Medium confidence"
