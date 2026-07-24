"""
splitter.py — DataSplitter for reproducible train/test dataset splitting in KiteML.
"""

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitter:
    """
    Manages reproducible train/test dataset splits.
    """

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
        task_type: str = "classification",
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split features and target into training and testing subsets.

        Parameters
        ----------
        X : pd.DataFrame
            Features.
        y : pd.Series
            Target.
        test_size : float
            Proportion of dataset to include in test split.
        random_state : int
            Random seed.
        task_type : str
            Task type for stratification.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            X_train, X_test, y_train, y_test.
        """
        stratify: Any = None
        if "classification" in task_type and y.nunique() > 1:
            stratify = y

        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
