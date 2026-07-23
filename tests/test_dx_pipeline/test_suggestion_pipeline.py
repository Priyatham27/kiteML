"""
test_suggestion_pipeline.py — Unit tests for SuggestionManager (Story 3.6).
"""

import pandas as pd
import pytest

from kiteml.pipeline.suggestion_manager import SuggestionManager


def test_suggestion_manager():
    mgr = SuggestionManager()
    df = pd.DataFrame({"const_col": [1, 1, 1], "target_col": [0, 1, 0]})

    suggs = mgr.generate(df)
    assert len(suggs) > 0
    formatted = mgr.format_suggestions(suggs)
    assert "KiteML Suggestions" in formatted
