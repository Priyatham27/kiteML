"""
orchestrator.py — UnifiedMLOrchestrator and train() root function for Story 5.8 in KiteML.

The train() function delegates to kiteml.core.train() which provides the full
rich Result API (typed metrics, validation, profiling, warnings, predict_proba, etc.)
while the ML DAG (MLWorkflowGraph) is available for advanced orchestration use.
"""

from typing import Any

import pandas as pd

from kiteml.config import (
    DEFAULT_CV_FOLDS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    DEFAULT_VERBOSE,
)
from kiteml.ml.context import MLContext
from kiteml.ml.report import UnifiedReport
from kiteml.ml.result import TrainingResult
from kiteml.ml.workflow_graph import MLWorkflowGraph


class UnifiedMLOrchestrator:
    """
    Master high-level orchestrator coordinating end-to-end ML training DAG.
    """

    def __init__(self) -> None:
        self.graph = MLWorkflowGraph()

    def train(
        self,
        dataframe: pd.DataFrame,
        target: str,
        optimize_hyperparameters: bool = False,
        policy: str = "balanced",
    ) -> TrainingResult:
        """
        Train end-to-end ML model pipeline on DataFrame using the ML DAG.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input pandas DataFrame.
        target : str
            Target column name.
        optimize_hyperparameters : bool
            Whether to run hyperparameter tuning.
        policy : str
            Model selection policy profile.

        Returns
        -------
        TrainingResult
            Result container containing best model, pipeline, prediction, save, and report.
        """
        if target not in dataframe.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame columns: {list(dataframe.columns)}")

        context = MLContext(dataframe=dataframe, target_column=target)
        executed_context = self.graph.execute(
            context,
            optimize_hyperparameters=optimize_hyperparameters,
            policy=policy,
        )

        bm = executed_context.best_model
        report = UnifiedReport(
            model_name=bm.model_name,
            problem_type=executed_context.problem_type,
            composite_score=bm.composite_score,
            explanation=bm.explanation,
            stage_timings=executed_context.metrics.get("stage_timings_sec", {}),
            total_time_sec=executed_context.metrics.get("total_execution_time_sec", 0.0),
        )

        return TrainingResult(
            best_model=bm,
            pipeline=executed_context.pipeline_result,
            problem_type=executed_context.problem_type,
            report=report,
            metrics=executed_context.metrics,
        )


def train(
    dataframe: pd.DataFrame | str,
    target: str | None = None,
    problem_type: str | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    scale: bool = True,
    random_state: int = DEFAULT_RANDOM_STATE,
    cv: int = DEFAULT_CV_FOLDS,
    verbose: bool = DEFAULT_VERBOSE,
    validate_data: bool = True,
    policy: str = "balanced",
    optimize_hyperparameters: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Flagship KiteML entry point — train the optimal ML model with one function call.

    Delegates to kiteml.core.train() to provide the full rich Result API:
    typed metrics, validation, data profiling, warnings, leaderboard, predict_proba, etc.

    Parameters
    ----------
    dataframe : pd.DataFrame or str
        Input dataset (DataFrame or path to CSV/Excel/JSON/Parquet).
    target : str, optional
        Target column name. Defaults to last column if not provided.
    problem_type : str, optional
        ``'classification'`` or ``'regression'``. Auto-detected when omitted.
    test_size : float
        Fraction of data for test set.
    scale : bool
        Apply StandardScaler inside Preprocessor.
    random_state : int
        Random seed for reproducibility.
    cv : int
        Number of cross-validation folds.
    verbose : bool
        Emit progress messages.
    validate_data : bool
        Run ValidationPipeline before training.
    policy : str
        Model selection policy (reserved for DAG orchestrator).
    optimize_hyperparameters : bool
        Whether to tune hyperparameters (reserved for DAG orchestrator).

    Returns
    -------
    Result
        Rich result with best model, typed metrics, report, validation, profiling etc.
    """
    from kiteml.core import train as _core_train

    return _core_train(
        data=dataframe,
        target=target,
        problem_type=problem_type,
        test_size=test_size,
        scale=scale,
        random_state=random_state,
        cv=cv,
        verbose=verbose,
        validate_data=validate_data,
    )
