"""
classification.py — Classification evaluation metrics and confusion matrix analytics for KiteML.
"""

import contextlib
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_proba: Any = None,
) -> dict[str, Any]:
    """
    Calculate classification evaluation metrics.

    Parameters
    ----------
    y_true : Any
        Ground truth labels.
    y_pred : Any
        Predicted labels.
    y_proba : Any, optional
        Predicted probability array.

    Returns
    -------
    dict[str, Any]
        Dictionary of classification metrics (accuracy, precision, recall, f1, roc_auc, confusion_matrix).
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    if not np.issubdtype(y_p.dtype, np.integer) and not np.issubdtype(y_p.dtype, np.bool_):
        y_p = np.round(y_p).astype(int)

    n_classes = len(np.unique(y_t))
    average = "binary" if n_classes <= 2 else "weighted"

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, average=average, zero_division=0))
    rec = float(recall_score(y_t, y_p, average=average, zero_division=0))
    f1 = float(f1_score(y_t, y_p, average=average, zero_division=0))

    auc = 0.0
    if y_proba is not None:
        with contextlib.suppress(Exception):
            if n_classes == 2:
                proba_col = y_proba[:, 1] if y_proba.ndim > 1 and y_proba.shape[1] > 1 else y_proba
                auc = float(roc_auc_score(y_t, proba_col))
            else:
                auc = float(roc_auc_score(y_t, y_proba, multi_class="ovr"))

    cm = confusion_matrix(y_t, y_p).tolist()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
    }
