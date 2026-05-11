"""
data_profiler.py — Orchestrates all intelligence modules into a DataProfile.

This is the central object stored on a Result after training.  It is the
single source of truth for all intelligence-layer findings.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from kiteml.intelligence.column_analyzer import ColumnAnalysisResult, analyze_columns
from kiteml.intelligence.schema_inference import DataSchema, infer_schema
from kiteml.intelligence.target_detection import TargetDetectionResult, detect_target
from kiteml.intelligence.problem_inference import ProblemInferenceResult, infer_problem_type_advanced
from kiteml.intelligence.quality_analyzer import QualityReport, analyze_quality
from kiteml.intelligence.imbalance_detector import ImbalanceReport, detect_imbalance
from kiteml.intelligence.outlier_detector import OutlierReport, detect_outliers
from kiteml.intelligence.leakage_detector import LeakageReport, detect_leakage
from kiteml.intelligence.correlation_analyzer import CorrelationReport, analyze_correlations
from kiteml.intelligence.cardinality_analyzer import CardinalityReport, analyze_cardinality
from kiteml.intelligence.text_detector import TextDetectionResult, detect_text_columns
from kiteml.intelligence.datetime_detector import DatetimeDetectionResult, detect_datetime_columns
from kiteml.intelligence.memory_optimizer import MemoryReport, analyze_memory
from kiteml.intelligence.feature_recommender import FeatureRecommendationReport, generate_recommendations
from kiteml.intelligence.recommendations import MasterRecommendationReport, build_recommendation_report


@dataclass
class DataProfile:
    """
    Complete intelligence profile of a dataset.

    Created once during kiteml.train() and stored on the Result object.
    All result.profile(), result.recommendations() etc. delegate here.
    """
    target: str
    problem_type: str

    # ── Intelligence layers ───────────────────────────────────────────────
    schema: DataSchema
    column_analysis: ColumnAnalysisResult
    target_detection: Optional[TargetDetectionResult]
    problem_inference: ProblemInferenceResult
    quality: QualityReport
    imbalance: Optional[ImbalanceReport]         # None for regression
    outliers: OutlierReport
    leakage: LeakageReport
    correlations: CorrelationReport
    cardinality: CardinalityReport
    text: TextDetectionResult
    datetime: DatetimeDetectionResult
    memory: MemoryReport
    feature_recommendations: FeatureRecommendationReport
    master_recommendations: MasterRecommendationReport


def build_data_profile(
    df: pd.DataFrame,
    target: str,
    problem_type: str,
) -> DataProfile:
    """
    Run all intelligence modules and assemble a DataProfile.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (including target column).
    target : str
        Name of the target column.
    problem_type : str
        ``'classification'`` or ``'regression'``.

    Returns
    -------
    DataProfile
    """
    # ── Core structural analysis ──────────────────────────────────────────
    schema          = infer_schema(df)
    col_analysis    = analyze_columns(df, exclude=[target])
    problem_inf     = infer_problem_type_advanced(df[target])

    # ── Quality & safety ─────────────────────────────────────────────────
    quality         = analyze_quality(df)
    leakage         = detect_leakage(df, target=target)
    outliers        = detect_outliers(df.drop(columns=[target], errors="ignore"))

    # ── Target-specific ───────────────────────────────────────────────────
    imbalance = None
    if problem_type == "classification":
        imbalance = detect_imbalance(df[target])

    # ── Feature intelligence ──────────────────────────────────────────────
    correlations    = analyze_correlations(df, target=target)
    cardinality     = analyze_cardinality(df.drop(columns=[target], errors="ignore"))
    text            = detect_text_columns(df.drop(columns=[target], errors="ignore"))
    datetime_info   = detect_datetime_columns(df.drop(columns=[target], errors="ignore"))
    memory          = analyze_memory(df)
    feat_recs       = generate_recommendations(df, col_analysis, target=target)

    # ── Master recommendations ────────────────────────────────────────────
    master = build_recommendation_report(
        quality=quality,
        leakage=leakage,
        imbalance=imbalance,
        outliers=outliers,
        features=feat_recs,
        correlations=correlations,
        cardinality=cardinality,
        memory=memory,
    )

    return DataProfile(
        target=target,
        problem_type=problem_type,
        schema=schema,
        column_analysis=col_analysis,
        target_detection=None,   # set by core.py when auto-detecting
        problem_inference=problem_inf,
        quality=quality,
        imbalance=imbalance,
        outliers=outliers,
        leakage=leakage,
        correlations=correlations,
        cardinality=cardinality,
        text=text,
        datetime=datetime_info,
        memory=memory,
        feature_recommendations=feat_recs,
        master_recommendations=master,
    )
