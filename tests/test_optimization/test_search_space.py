"""
test_search_space.py — Unit tests for SearchSpace (Story 5.3).
"""

import pytest

from kiteml.optimization import SearchSpace


def test_search_space_generation():
    ss = SearchSpace.get_default_search_space("RandomForestClassifier")
    assert "n_estimators" in ss.param_grid
    assert ss.total_combinations() > 0
