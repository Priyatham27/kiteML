"""
trainer.py — Final model training engine.

Responsibility
--------------
This module has *one job*: fit the selected model on the full training set.

It is intentionally thin.  Keeping training isolated from selection and
evaluation means future changes (distributed training, GPU, incremental
learning) only touch this file.

Why not just call model.fit() in core.py?
-----------------------------------------
Separation of concerns.  core.py is the orchestrator; trainer.py is the
execution layer.  This pattern mirrors scikit-learn's own Pipeline design
and makes each component unit-testable in isolation.

Time Tracking
-------------
Training time is measured with time.perf_counter() (highest available
resolution) and returned alongside the fitted model so core.py can include
it in the Result and surface it to the user.  This is surfaced at INFO
level so users always see how long training took without needing verbose
debug output.
"""

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def train_model(
    model: Any,
    X_train: np.ndarray,
    y_train: Any,
) -> tuple[Any, float]:
    """
    Fit the model on the provided training data.

    Parameters
    ----------
    model : estimator
        Any scikit-learn compatible *unfitted* model instance.
    X_train : array-like of shape (n_samples, n_features)
        Processed training feature matrix (output of Preprocessor).
    y_train : array-like of shape (n_samples,)
        Training target vector.

    Returns
    -------
    model : estimator
        The *fitted* model (mutated in-place by sklearn convention, but
        also returned for explicit assignment clarity).
    training_time : float
        Wall-clock seconds the ``model.fit()`` call took, measured with
        ``time.perf_counter()`` for sub-millisecond accuracy.

    Notes
    -----
    * Training time is logged at INFO level so it is always visible to
      the user without needing to enable DEBUG logging.
    * The function does **not** perform any splitting or preprocessing —
      that is core.py's responsibility.
    """
    model_name = type(model).__name__
    n_samples = len(y_train) if hasattr(y_train, "__len__") else "?"

    logger.debug("⚙️  Fitting %s on %s samples…", model_name, n_samples)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - t0

    logger.info("⏱️  %s trained in %.2fs", model_name, training_time)

    return model, training_time
