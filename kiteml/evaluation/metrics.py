"""
metrics.py — Standardised model evaluation metrics for KiteML.

Design
------
* evaluate_model() is the single entry point for both classification and
  regression.  All other KiteML components consume its output dict.
* Keys are stable across versions — downstream code can rely on them.
* Classification metrics use ``average='weighted'`` to handle class
  imbalance gracefully without crashing on rare classes.
* ``zero_division=0`` prevents warnings when a class has no predictions.

Classification metric keys
--------------------------
    "accuracy"               float   overall correctness
    "precision"              float   weighted precision
    "recall"                 float   weighted recall
    "f1_score"               float   weighted F1
    "confusion_matrix"       list    nested list (int values)
    "classification_report"  str     full sklearn text report

Regression metric keys
----------------------
    "r2_score"   float   coefficient of determination (fit quality)
    "mse"        float   mean squared error
    "rmse"       float   root mean squared error (primary sort metric)
    "mae"        float   mean absolute error
"""

import logging
from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    X_test: Any,
    y_test: Any,
    problem_type: str = "classification",
) -> Dict[str, Any]:
    """
    Evaluate a trained model and return a standardised metrics dictionary.

    Parameters
    ----------
    model : estimator
        A **fitted** scikit-learn compatible model.
    X_test : array-like of shape (n_samples, n_features)
        Processed test feature matrix (output of Preprocessor.transform).
    y_test : array-like of shape (n_samples,)
        True target values for the test set.
    problem_type : str
        ``'classification'`` or ``'regression'``.

    Returns
    -------
    dict
        Flat dictionary of metrics.  See module docstring for all keys.

    Raises
    ------
    ValueError
        If ``problem_type`` is not recognised.
    """
    if problem_type not in ("classification", "regression"):
        raise ValueError(f"Unknown problem_type '{problem_type}'. " "Expected 'classification' or 'regression'.")

    y_pred = model.predict(X_test)

    if problem_type == "classification":
        metrics: Dict[str, Any] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, zero_division=0),
        }
    else:  # regression
        mse = float(mean_squared_error(y_test, y_pred))
        metrics = {
            "r2_score": float(r2_score(y_test, y_pred)),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
        }

    logger.debug("Evaluation complete: %s", {k: v for k, v in metrics.items() if isinstance(v, float)})
    return metrics
