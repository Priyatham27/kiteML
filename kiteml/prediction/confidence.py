"""
confidence.py — ConfidenceEngine for extracting prediction confidence and probabilities in KiteML.
"""

import contextlib
from typing import Any

import numpy as np


class ConfidenceEngine:
    """
    Computes prediction probabilities and confidence scores.
    """

    def compute_confidence(self, model: Any, X: Any) -> tuple[Any | None, np.ndarray | None]:
        """
        Compute probabilities and row-level confidence scores.

        Parameters
        ----------
        model : Any
            Fitted estimator model.
        X : Any
            Feature matrix.

        Returns
        -------
        tuple[Any | None, np.ndarray | None]
            Probabilities matrix and confidence array.
        """
        if not hasattr(model, "predict_proba"):
            return None, None

        try:
            proba = model.predict_proba(X)
            confidence = np.max(proba, axis=1)
            return proba, confidence
        except Exception:
            return None, None
