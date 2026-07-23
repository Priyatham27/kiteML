"""
test_engine.py — Unit tests for SuggestionEngine (Story 3.5).
"""

import pandas as pd
import pytest

from kiteml.suggestions.engine import SuggestionEngine


def test_suggestion_engine_generation():
    engine = SuggestionEngine(min_confidence=0.5)
    df = pd.DataFrame(
        {
            "const_feature": [5, 5, 5, 5, 5],
            "price": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )

    suggestions = engine.generate({"df": df, "target": "prcie", "available_columns": ["const_feature", "price"]})

    assert len(suggestions) > 0
    # Top suggestion should be column typo correction or schema optimization
    titles = [s.title for s in suggestions]
    assert any("price" in t.lower() or "const_feature" in t for t in titles)
