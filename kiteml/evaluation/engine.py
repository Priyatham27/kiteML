"""
engine.py — EvaluationEngine and EvaluationResult master entry point for KiteML model evaluation.
"""

import contextlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from kiteml.evaluation.benchmark import BenchmarkEngine, BenchmarkMetrics
from kiteml.evaluation.classification import evaluate_classification_metrics
from kiteml.evaluation.composite import CompositeScorer
from kiteml.evaluation.diagnostics import DiagnosticsEngine
from kiteml.evaluation.regression import evaluate_regression_metrics
from kiteml.evaluation.report import EvaluationReport


@dataclass
class EvaluationResult:
    """
    Result container returned by EvaluationEngine.evaluate().
    """

    report: EvaluationReport
    composite_score: float
    metrics: dict[str, Any]
    benchmark: BenchmarkMetrics


class EvaluationEngine:
    """
    Master Intelligent Model Evaluation Engine for evaluating fitted ML models.
    """

    def __init__(self) -> None:
        self.benchmark_engine = BenchmarkEngine()
        self.diagnostics_engine = DiagnosticsEngine()
        self.composite_scorer = CompositeScorer()

    def evaluate(
        self,
        model: Any,
        X_test: pd.DataFrame | Any,
        y_test: pd.Series | Any,
        problem_type: str | None = None,
    ) -> EvaluationResult:
        """
        Evaluate a fitted model on hold-out test set.

        Parameters
        ----------
        model : Any
            Fitted estimator model.
        X_test : pd.DataFrame | Any
            Test feature matrix.
        y_test : pd.Series | Any
            Test ground truth targets.
        problem_type : str, optional
            Explicit task type ('classification' or 'regression').

        Returns
        -------
        EvaluationResult
            Complete evaluation result container.
        """
        task_type = problem_type or (
            "regression" if np.issubdtype(np.asarray(y_test).dtype, np.floating) else "classification"
        )

        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            with contextlib.suppress(Exception):
                y_proba = model.predict_proba(X_test)

        if "regression" in task_type:
            metrics = evaluate_regression_metrics(y_test, y_pred)
        else:
            metrics = evaluate_classification_metrics(y_test, y_pred, y_proba=y_proba)

        benchmark = self.benchmark_engine.benchmark_model(model, X_test)
        diagnostics = self.diagnostics_engine.diagnose(y_test, y_pred, task_type=task_type)
        comp_score = self.composite_scorer.calculate_composite_score(metrics, benchmark, task_type=task_type)

        report = EvaluationReport(
            task_type=task_type,
            composite_score=comp_score,
            metrics=metrics,
            benchmark=benchmark,
            diagnostics=diagnostics,
        )

        return EvaluationResult(
            report=report,
            composite_score=comp_score,
            metrics=metrics,
            benchmark=benchmark,
        )
