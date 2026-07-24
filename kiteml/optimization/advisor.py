"""
advisor.py — OptimizationAdvisor Flagship Adaptive Strategy Selection Feature for KiteML.
"""

from kiteml.optimization.search_space import SearchSpace
from kiteml.optimization.strategies import (
    GridSearchStrategy,
    OptimizationStrategy,
    RandomSearchStrategy,
)


class OptimizationAdvisor:
    """
    Intelligently inspects search space cardinality and dataset scale to choose optimal strategy.
    """

    def select_strategy(
        self,
        search_space: SearchSpace,
        n_samples: int = 1000,
        max_trials: int = 10,
    ) -> tuple[OptimizationStrategy, str]:
        """
        Select best optimization strategy based on space size and dataset characteristics.

        Parameters
        ----------
        search_space : SearchSpace
            Search space definition.
        n_samples : int
            Dataset sample size.
        max_trials : int
            Trial limit.

        Returns
        -------
        tuple[OptimizationStrategy, str]
            Selected strategy instance and human-readable explanation.
        """
        total_combos = search_space.total_combinations()

        if total_combos == 0:
            return GridSearchStrategy(), "Empty search space; using default single grid trial."

        if total_combos <= 12 and n_samples < 5000:
            return GridSearchStrategy(), f"Small search space ({total_combos} combinations); selecting Grid Search."

        return RandomSearchStrategy(), f"Large search space ({total_combos} combinations); selecting Random Search."
