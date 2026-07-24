"""
cross_validation.py — CrossValidationEngine for creating cross-validation splitters in KiteML.
"""

from typing import Any

from sklearn.model_selection import KFold, StratifiedKFold


class CrossValidationEngine:
    """
    Factory generating appropriate scikit-learn cross-validation splitters.
    """

    def get_cv(self, task_type: str, n_splits: int = 5, random_state: int = 42) -> Any:
        """
        Get cross-validation splitter.

        Parameters
        ----------
        task_type : str
            ML task type.
        n_splits : int
            Number of folds.
        random_state : int
            Random seed.

        Returns
        -------
        BaseCrossValidator
            Configured cross-validation splitter.
        """
        if "classification" in task_type:
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
