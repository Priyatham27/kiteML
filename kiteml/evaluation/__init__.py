"""
evaluation/ — Intelligent Model Evaluation Engine package for KiteML.
"""

from kiteml.evaluation.benchmark import BenchmarkEngine, BenchmarkMetrics
from kiteml.evaluation.classification import evaluate_classification_metrics
from kiteml.evaluation.composite import CompositeScorer
from kiteml.evaluation.context import EvaluationContext
from kiteml.evaluation.diagnostics import DiagnosticsEngine
from kiteml.evaluation.engine import EvaluationEngine, EvaluationResult
from kiteml.evaluation.regression import evaluate_regression_metrics
from kiteml.evaluation.report import EvaluationReport, generate_report

__all__ = [
    "EvaluationEngine",
    "EvaluationResult",
    "EvaluationReport",
    "generate_report",
    "EvaluationContext",
    "CompositeScorer",
    "BenchmarkEngine",
    "BenchmarkMetrics",
    "DiagnosticsEngine",
    "evaluate_regression_metrics",
    "evaluate_classification_metrics",
]
