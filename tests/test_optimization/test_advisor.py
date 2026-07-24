"""
test_advisor.py — Unit tests for OptimizationAdvisor (Story 5.3 Flagship Feature).
"""

import pytest

from kiteml.optimization import (
    GridSearchStrategy,
    OptimizationAdvisor,
    RandomSearchStrategy,
    SearchSpace,
)


def test_advisor_strategy_selection():
    advisor = OptimizationAdvisor()

    small_ss = SearchSpace(param_grid={"a": [1, 2], "b": [3, 4]})
    strat, _ = advisor.select_strategy(small_ss, n_samples=100)
    assert isinstance(strat, GridSearchStrategy)

    large_ss = SearchSpace(param_grid={"a": list(range(10)), "b": list(range(10))})
    strat2, _ = advisor.select_strategy(large_ss, n_samples=100)
    assert isinstance(strat2, RandomSearchStrategy)
