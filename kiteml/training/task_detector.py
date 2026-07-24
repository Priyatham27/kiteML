"""
task_detector.py — Automatic ML task detection engine for KiteML training.
"""

import pandas as pd
from pandas.api.types import is_numeric_dtype


class TaskDetector:
    """
    Automatically infers machine learning task type from target Series properties.
    """

    def detect_task(self, target_series: pd.Series) -> str:
        """
        Detect ML task type.

        Parameters
        ----------
        target_series : pd.Series
            Target feature values.

        Returns
        -------
        str
            'regression', 'binary_classification', or 'multiclass_classification'.
        """
        if target_series.empty:
            raise ValueError("Cannot detect task type from empty target Series.")

        clean_target = target_series.dropna()
        n_unique = clean_target.nunique()

        if n_unique == 2:
            return "binary_classification"

        if is_numeric_dtype(clean_target):
            # Floating point or high cardinality numeric indicates regression
            if clean_target.dtype in ["float64", "float32"] or n_unique > 20:
                return "regression"

        return "multiclass_classification"
