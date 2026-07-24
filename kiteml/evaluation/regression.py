"""
regression.py — Regression evaluation metrics and residual analytics for KiteML.
"""

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """
    Calculate comprehensive regression evaluation metrics.

    Parameters
    ----------
    y_true : Any
        Ground truth target values.
    y_pred : Any
        Predicted target values.

    Returns
    -------
    dict[str, float]
        Dictionary of regression metrics (mae, mse, rmse, r2, median_absolute_error).
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    mae = float(mean_absolute_error(y_t, y_p))
    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_t, y_p))
    med_ae = float(np.median(np.abs(y_t - y_p)))

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "median_absolute_error": med_ae,
    }
