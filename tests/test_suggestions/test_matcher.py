"""
test_matcher.py — Unit tests for matcher algorithms (Story 3.5).
"""

import pytest

from kiteml.suggestions.matcher import (
    levenshtein_distance,
    match_column_name,
    string_similarity,
)


def test_levenshtein_distance():
    assert levenshtein_distance("price", "price") == 0
    assert levenshtein_distance("prcie", "price") == 2
    assert levenshtein_distance("cat", "hat") == 1


def test_string_similarity():
    assert string_similarity("price", "price") == 1.0
    assert string_similarity("Price", "price") == 0.98
    assert string_similarity("prcie", "price") >= 0.6
    assert string_similarity("unrelated", "price") < 0.4


def test_match_column_name():
    cols = ["Price", "SellingPrice", "UnitPrice", "Age", "City"]
    matches = match_column_name("prcie", cols)

    assert len(matches) > 0
    top_col, top_score = matches[0]
    assert top_col == "Price"
    assert top_score >= 0.6
