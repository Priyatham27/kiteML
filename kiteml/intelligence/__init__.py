"""
intelligence/ — KiteML Phase 2 Intelligence Layer.

This package transforms KiteML from a model automation tool into an
intelligent ML assistant that understands data, detects issues,
infers intent, and generates actionable insights.

Public API
----------
Each sub-module is independently importable and returns structured objects.
No module depends on another within this package (star-topology design).
"""

from kiteml.intelligence.column_analyzer import analyze_columns, ColumnType
from kiteml.intelligence.schema_inference import infer_schema
from kiteml.intelligence.target_detection import detect_target
from kiteml.intelligence.problem_inference import infer_problem_type_advanced
from kiteml.intelligence.quality_analyzer import analyze_quality
from kiteml.intelligence.imbalance_detector import detect_imbalance
from kiteml.intelligence.outlier_detector import detect_outliers
from kiteml.intelligence.leakage_detector import detect_leakage
from kiteml.intelligence.correlation_analyzer import analyze_correlations
from kiteml.intelligence.cardinality_analyzer import analyze_cardinality
from kiteml.intelligence.text_detector import detect_text_columns
from kiteml.intelligence.datetime_detector import detect_datetime_columns
from kiteml.intelligence.memory_optimizer import analyze_memory
from kiteml.intelligence.feature_recommender import generate_recommendations
from kiteml.intelligence.explainability import explain_model
from kiteml.intelligence.recommendations import build_recommendation_report

__all__ = [
    "analyze_columns", "ColumnType",
    "infer_schema",
    "detect_target",
    "infer_problem_type_advanced",
    "analyze_quality",
    "detect_imbalance",
    "detect_outliers",
    "detect_leakage",
    "analyze_correlations",
    "analyze_cardinality",
    "detect_text_columns",
    "detect_datetime_columns",
    "analyze_memory",
    "generate_recommendations",
    "explain_model",
    "build_recommendation_report",
]
