"""
strategies.py — OptimizationStrategy base class with Grid and Random Search implementations.
"""

import itertools
import random
from abc import ABC, abstractmethod
from typing import Any

from kiteml.optimization.search_space import SearchSpace


class OptimizationStrategy(ABC):
    """
    Abstract base class for hyperparameter sampling strategies.
    """

    @abstractmethod
    def generate_parameter_combinations(
        self,
        search_space: SearchSpace,
        max_trials: int = 10,
        random_state: int = 42,
    ) -> list[dict[str, Any]]:
        """Generate parameter trial configurations."""
        pass


class GridSearchStrategy(OptimizationStrategy):
    """
    Exhaustive grid search strategy.
    """

    def generate_parameter_combinations(
        self,
        search_space: SearchSpace,
        max_trials: int = 10,
        random_state: int = 42,
    ) -> list[dict[str, Any]]:
        if not search_space.param_grid:
            return [{}]

        keys = list(search_space.param_grid.keys())
        values = list(search_space.param_grid.values())
        combos = [dict(zip(keys, v, strict=False)) for v in itertools.product(*values)]
        return combos[:max_trials]


class RandomSearchStrategy(OptimizationStrategy):
    """
    Stochastic random sampling strategy.
    """

    def generate_parameter_combinations(
        self,
        search_space: SearchSpace,
        max_trials: int = 10,
        random_state: int = 42,
    ) -> list[dict[str, Any]]:
        grid_strategy = GridSearchStrategy()
        all_combos = grid_strategy.generate_parameter_combinations(search_space, max_trials=1000)
        if not all_combos or all_combos == [{}]:
            return [{}]

        rng = random.Random(random_state)
        sample_size = min(max_trials, len(all_combos))
        return rng.sample(all_combos, k=sample_size)
