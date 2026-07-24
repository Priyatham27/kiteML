"""
test_strategies.py — Unit tests for OptimizationStrategies (Story 5.3).
"""

import pytest

from kiteml.optimization import GridSearchStrategy, RandomSearchStrategy, SearchSpace


def test_grid_search_strategy():
    ss = SearchSpace(param_grid={"a": [1, 2], "b": [10, 20]})
    combos = GridSearchStrategy().generate_parameter_combinations(ss)
    assert len(combos) == 4


def test_random_search_strategy():
    ss = SearchSpace(param_grid={"a": list(range(10)), "b": list(range(10))})
    combos = RandomSearchStrategy().generate_parameter_combinations(ss, max_trials=5)
    assert len(combos) == 5
