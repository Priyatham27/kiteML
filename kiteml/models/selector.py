"""
selector.py — Automatic model selection via cross-validation.

Design
------
* Receives *training data only* — the train/test split is done upstream
  in core.py.  This module never sees test data → zero leakage.
* Scores every registered model with k-fold CV on the training set.
* Returns:
    - best_model  : the unfitted estimator that scored highest (ready for
                    final .fit() in the trainer)
    - all_results : structured dict  →  {model_name: {"score": float, "rank": int}}
                    Errors are captured per-model so one bad model never
                    crashes the whole selection process.

Scoring convention
------------------
    Classification  →  accuracy   (higher = better)
    Regression      →  R²         (higher = better)

The "score" field in all_results always follows the "higher = better"
convention.  The report layer converts R² to RMSE for display purposes.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from kiteml.config import DEFAULT_CV_FOLDS, DEFAULT_N_JOBS, DEFAULT_RANDOM_STATE
from kiteml.models.registry import get_classification_models, get_regression_models

logger = logging.getLogger(__name__)


def select_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str = "classification",
    random_state: int = DEFAULT_RANDOM_STATE,
    cv: int = DEFAULT_CV_FOLDS,
) -> Tuple[Any, Dict[str, Dict[str, Any]]]:
    """
    Select the best model using k-fold cross-validation on training data.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training feature matrix — **no test data should be passed here**.
    y_train : array-like of shape (n_samples,)
        Training target vector.
    problem_type : str
        ``'classification'`` or ``'regression'``.
    random_state : int
        Forwarded to models that accept it; also seeds CV shuffling.
    cv : int
        Number of cross-validation folds. Default ``5``.

    Returns
    -------
    best_model : estimator
        The *unfitted* estimator instance with the highest mean CV score.
        core.py is responsible for the final ``model.fit(X_train, y_train)``.
    all_results : dict
        ``{model_name: {"score": float, "rank": int, "error": str|None}}``

        - ``score``  — mean CV score (accuracy for classification, R² for
                        regression). ``None`` when an error occurred.
        - ``rank``   — 1-indexed rank (1 = best). ``None`` on error.
        - ``error``  — exception message string, or ``None`` on success.

    Raises
    ------
    RuntimeError
        If *all* candidate models fail during cross-validation.
    """
    # ── Load candidate models from registry ──────────────────────────────
    if problem_type == "classification":
        models = get_classification_models()
        scoring = "accuracy"
    elif problem_type == "regression":
        models = get_regression_models()
        scoring = "r2"
    else:
        raise ValueError(
            f"Unknown problem_type '{problem_type}'. "
            "Expected 'classification' or 'regression'."
        )

    # ── Cross-validate every candidate ───────────────────────────────────
    raw_scores: Dict[str, Optional[float]] = {}
    errors: Dict[str, Optional[str]] = {}

    for name, model in models.items():
        try:
            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=DEFAULT_N_JOBS,
            )
            mean_score = float(scores.mean())
            raw_scores[name] = mean_score
            errors[name] = None
            logger.debug("  %s → %.4f (±%.4f)", name, mean_score, scores.std())
        except Exception as exc:                       # noqa: BLE001
            raw_scores[name] = None
            errors[name] = str(exc)
            logger.warning("  %s → FAILED: %s", name, exc)

    # ── Rank models by score (higher = better) ───────────────────────────
    scored_models = {k: v for k, v in raw_scores.items() if v is not None}

    if not scored_models:
        raise RuntimeError(
            "All candidate models failed during cross-validation. "
            "Check your data for issues (infinite values, too few samples, etc.)."
        )

    sorted_names = sorted(scored_models, key=lambda k: scored_models[k], reverse=True)

    # ── Build structured all_results ─────────────────────────────────────
    all_results: Dict[str, Dict[str, Any]] = {}
    for rank, name in enumerate(sorted_names, start=1):
        all_results[name] = {
            "score": scored_models[name],
            "rank": rank,
            "error": None,
        }
    for name in errors:
        if errors[name] is not None:                   # failed models
            all_results[name] = {
                "score": None,
                "rank": None,
                "error": errors[name],
            }

    # ── Select best model ─────────────────────────────────────────────────
    best_name = sorted_names[0]
    best_model = models[best_name]                     # unfitted instance

    logger.info(
        "🏆 Best model: %s  (CV %s = %.4f)",
        best_name,
        scoring,
        scored_models[best_name],
    )

    return best_model, all_results
