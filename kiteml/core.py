"""
core.py - Main entry point for KiteML.

Provides the `train` function that orchestrates the full ML pipeline:
  data loading → validation → split → Preprocessor (fit on train, transform test)
  → model selection → training → evaluation → Result.

Key design decisions
--------------------
* Split happens FIRST — the Preprocessor is fit only on training data,
  eliminating any risk of data leakage.
* A single Preprocessor object (backed by sklearn Pipeline internals) replaces
  the separate encoder + scaler steps, making the pipeline cleaner and the
  Result self-contained.
* Feature names come from Preprocessor.feature_names so importance values
  correctly map to OHE-expanded column names.
* A single canonical train/test split is created here and shared everywhere;
  select_best_model and train_model never re-split.
* All defaults (random_state, test_size, cv folds) come from kiteml.config
  so changing a global default is a one-file edit.
* Logging (stdlib) is used instead of bare print() so callers can control
  verbosity via the standard logging framework.
* Training time is measured in trainer.py and threaded back through core.py
  into the Result's TrainingTimes object for full per-phase visibility.
"""

import logging
import time
from typing import Optional, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from kiteml.config import (
    DEFAULT_CV_FOLDS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    DEFAULT_VERBOSE,
)
from kiteml.evaluation.metrics import evaluate_model
from kiteml.evaluation.report import generate_report
from kiteml.intelligence.data_profiler import build_data_profile
from kiteml.models.selector import select_best_model
from kiteml.output.result import Result
from kiteml.preprocessing.pipeline import Preprocessor
from kiteml.training.trainer import train_model
from kiteml.utils.data_loader import load_data
from kiteml.utils.type_inference import infer_problem_type

logger = logging.getLogger(__name__)


def _setup_utf8_logging():
    """Configure logging with a UTF-8 safe handler for Windows compatibility."""
    import io
    import sys

    try:
        # Try to reconfigure stdout to UTF-8
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    handler = logging.StreamHandler(
        stream=(
            io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            if hasattr(sys.stdout, "buffer")
            else sys.stdout
        )
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False


def train(
    data: Union[str, pd.DataFrame],
    target: Optional[str] = None,
    problem_type: Optional[str] = None,
    test_size: float = DEFAULT_TEST_SIZE,
    scale: bool = True,
    random_state: int = DEFAULT_RANDOM_STATE,
    cv: int = DEFAULT_CV_FOLDS,
    verbose: bool = DEFAULT_VERBOSE,
) -> Result:
    """
    Train and evaluate ML models on the given dataset.

    Parameters
    ----------
    data : str or pd.DataFrame
        Path to a CSV / Excel / JSON / Parquet file, or a pandas DataFrame.
    target : str, optional
        Name of the target column. Defaults to the last column if not provided.
    problem_type : str, optional
        ``'classification'`` or ``'regression'``. Auto-detected when omitted.
    test_size : float
        Fraction of data reserved for the test set.
        Default from ``kiteml.config.DEFAULT_TEST_SIZE`` (0.2).
    scale : bool
        Apply StandardScaler inside the Preprocessor. Default ``True``.
        (Passed for API compatibility; Preprocessor always scales internally.)
    random_state : int
        Random seed for reproducibility.
        Default from ``kiteml.config.DEFAULT_RANDOM_STATE`` (42).
    cv : int
        Number of cross-validation folds used during model selection.
        Default from ``kiteml.config.DEFAULT_CV_FOLDS`` (5).
    verbose : bool
        Emit progress messages via the ``kiteml.core`` logger. Default ``True``.

    Returns
    -------
    Result
        A :class:`~kiteml.output.result.Result` containing the best model,
        typed metrics (ClassificationMetrics or RegressionMetrics), report,
        feature importances, fitted Preprocessor, and per-phase timings.

    Raises
    ------
    ValueError
        If the dataset is empty, the target column is missing, or
        ``problem_type`` is an unrecognised string.
    """
    # ------------------------------------------------------------------ #
    # Configure logging level for this run                                 #
    # ------------------------------------------------------------------ #
    if verbose:
        logging.basicConfig(format="%(message)s", level=logging.INFO)
        _setup_utf8_logging()

    total_start = time.perf_counter()

    # ------------------------------------------------------------------ #
    # Step 1 – Load data                                                   #
    # ------------------------------------------------------------------ #
    logger.info("📦 Loading data...")
    df = load_data(data)

    # ------------------------------------------------------------------ #
    # Step 2 – Validate inputs                                             #
    # ------------------------------------------------------------------ #
    if df.empty:
        raise ValueError("The dataset is empty.")

    if target is None:
        target = df.columns[-1]
        logger.info(f"ℹ️  No target specified — using last column: '{target}'")

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found in the dataset. " f"Available columns: {list(df.columns)}"
        )

    if problem_type is not None and problem_type not in ("classification", "regression"):
        raise ValueError(
            f"Invalid problem_type '{problem_type}'. "
            "Choose 'classification', 'regression', or leave as None for auto-detection."
        )

    logger.info(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} cols")

    # ------------------------------------------------------------------ #
    # Step 3 – Infer problem type                                          #
    # ------------------------------------------------------------------ #
    if problem_type is None:
        problem_type = infer_problem_type(df, target)
    logger.info(f"🔍 Problem type: {problem_type}")

    # ------------------------------------------------------------------ #
    # Step 4 – Split BEFORE any learnable preprocessing (no leakage)       #
    # ------------------------------------------------------------------ #
    logger.info(f"✂️  Splitting data (test_size={test_size}, seed={random_state})...")
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # ------------------------------------------------------------------ #
    # Step 5 – Fit Preprocessor on training data, transform both splits    #
    # (Preprocessor is backed internally by an sklearn Pipeline)           #
    # ------------------------------------------------------------------ #
    logger.info("🔧 Preprocessing (impute → encode → scale)...")
    preprocessor = Preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.feature_names
    logger.info(f"   → {len(feature_names)} features after encoding")

    # ------------------------------------------------------------------ #
    # Step 6 – Select best model using training data only (via CV)         #
    # ------------------------------------------------------------------ #
    logger.info(f"🤖 Selecting best model (CV={cv} folds)...")
    best_model, all_results = select_best_model(
        X_train_processed,
        y_train,
        problem_type=problem_type,
        random_state=random_state,
        cv=cv,
    )

    # ------------------------------------------------------------------ #
    # Step 7 – Train final model on full training set (via trainer)        #
    # trainer returns (fitted_model, training_time_seconds)                #
    # ------------------------------------------------------------------ #
    best_model_name = type(best_model).__name__
    logger.info(f"🏋️  Training {best_model_name} on full training set...")
    best_model, training_time = train_model(best_model, X_train_processed, y_train)

    # ------------------------------------------------------------------ #
    # Step 8 – Evaluate on held-out test set                               #
    # ------------------------------------------------------------------ #
    logger.info("📊 Evaluating on test set...")
    metrics = evaluate_model(best_model, X_test_processed, y_test, problem_type=problem_type)

    # ------------------------------------------------------------------ #
    # Step 9 – Extract feature importances mapped to real feature names     #
    # ------------------------------------------------------------------ #
    feature_importances = None
    if hasattr(best_model, "feature_importances_") and feature_names:
        feature_importances = dict(zip(feature_names, best_model.feature_importances_))
    elif hasattr(best_model, "coef_") and feature_names:
        coef = best_model.coef_
        if coef.ndim > 1:
            coef = coef[0]  # first row for multi-class
        feature_importances = dict(zip(feature_names, coef))

    # ------------------------------------------------------------------ #
    # Step 10 – Generate human-readable report                             #
    # ------------------------------------------------------------------ #
    report = generate_report(
        metrics,
        problem_type=problem_type,
        model_name=best_model_name,
        all_results=all_results,
    )
    logger.info("\n" + report)

    total_elapsed = time.perf_counter() - total_start
    logger.info(f"✅ Done in {total_elapsed:.2f}s  (training: {training_time:.2f}s)")

    # ------------------------------------------------------------------ #
    # Step 11 — Build DataProfile (Phase 2 intelligence)                  #
    # ------------------------------------------------------------------ #
    logger.info("🧠 Building dataset intelligence profile...")
    try:
        data_profile = build_data_profile(df, target=target, problem_type=problem_type)
    except Exception as exc:
        logger.warning("⚠️  Profiling skipped due to error: %s", exc)
        data_profile = None

    # ------------------------------------------------------------------ #
    # Step 12 — Build and return Result                                   #
    # ------------------------------------------------------------------ #
    result = Result(
        model=best_model,
        model_name=best_model_name,
        metrics=metrics,
        report=report,
        problem_type=problem_type,
        all_results=all_results,
        preprocessor=preprocessor,
        feature_importances=feature_importances,
        feature_names=feature_names,
        elapsed_time=total_elapsed,
        training_time=training_time,
        data_profile=data_profile,
    )

    return result
