"""
search_space.py — SearchSpace parameter generator for hyperparameter tuning in KiteML.
"""

from typing import Any


class SearchSpace:
    """
    Defines hyperparameter search spaces and provides default parameter grids for model algorithms.
    """

    def __init__(self, param_grid: dict[str, list[Any]] | None = None) -> None:
        self.param_grid = param_grid or {}

    @classmethod
    def get_default_search_space(cls, model_name: str) -> "SearchSpace":
        """
        Extract default search space for a given algorithm model name.

        Parameters
        ----------
        model_name : str
            Registered model name.

        Returns
        -------
        SearchSpace
            Populated SearchSpace instance.
        """
        grid: dict[str, list[Any]] = {}

        if "RandomForest" in model_name or "ExtraTrees" in model_name:
            grid = {
                "n_estimators": [20, 50, 100],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
            }
        elif "GradientBoosting" in model_name:
            grid = {
                "n_estimators": [20, 50],
                "learning_rate": [0.05, 0.1, 0.2],
            }
        elif "LogisticRegression" in model_name or "SVC" in model_name:
            grid = {
                "C": [0.01, 0.1, 1.0, 10.0],
            }
        elif model_name in ["Ridge", "Lasso", "ElasticNet"]:
            grid = {
                "alpha": [0.01, 0.1, 1.0, 10.0],
            }
        elif "KNeighbors" in model_name:
            grid = {
                "n_neighbors": [3, 5, 7],
            }
        else:
            grid = {}

        return cls(param_grid=grid)

    def total_combinations(self) -> int:
        """Calculate total number of discrete parameter combinations."""
        if not self.param_grid:
            return 0
        total = 1
        for vals in self.param_grid.values():
            total *= len(vals)
        return total
