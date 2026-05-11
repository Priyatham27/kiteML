"""
feature_recommender.py — Automatic feature engineering recommendations.

Aggregates signals from column analysis to suggest concrete, actionable
preprocessing improvements before or after training.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from kiteml.intelligence.column_analyzer import ColumnAnalysisResult, ColumnType


@dataclass
class FeatureRecommendation:
    """A single actionable feature preprocessing recommendation."""
    column: str
    action: str          # "drop" | "encode" | "log_transform" | "extract" | "flag" | "inspect"
    reason: str
    priority: str        # "high" | "medium" | "low"
    impact: str          # human description of expected impact


@dataclass
class FeatureRecommendationReport:
    """All feature recommendations for a dataset."""
    recommendations: List[FeatureRecommendation]
    drop_candidates: List[str]
    encode_candidates: List[str]
    transform_candidates: List[str]
    summary: str

    def high_priority(self) -> List[FeatureRecommendation]:
        return [r for r in self.recommendations if r.priority == "high"]


def generate_recommendations(
    df: pd.DataFrame,
    column_analysis: ColumnAnalysisResult,
    target: Optional[str] = None,
) -> FeatureRecommendationReport:
    """
    Generate feature engineering recommendations from column analysis.

    Parameters
    ----------
    df : pd.DataFrame
    column_analysis : ColumnAnalysisResult
        Output of analyze_columns().
    target : str, optional
        Target column name to exclude.

    Returns
    -------
    FeatureRecommendationReport
    """
    recs: List[FeatureRecommendation] = []
    drop_cands: List[str] = []
    encode_cands: List[str] = []
    transform_cands: List[str] = []

    for col, profile in column_analysis.profiles.items():
        if col == target:
            continue

        # ── IDENTIFIER → drop ─────────────────────────────────────────────
        if profile.column_type == ColumnType.IDENTIFIER:
            recs.append(FeatureRecommendation(
                column=col, action="drop",
                reason=f"'{col}' appears to be an identifier (unique_ratio={profile.unique_ratio:.2%}). IDs carry no predictive value.",
                priority="high",
                impact="Reduces noise and prevents model from memorizing row IDs.",
            ))
            drop_cands.append(col)

        # ── CONSTANT → drop ───────────────────────────────────────────────
        elif profile.column_type == ColumnType.CONSTANT:
            recs.append(FeatureRecommendation(
                column=col, action="drop",
                reason=f"'{col}' has only one unique value — zero variance.",
                priority="high",
                impact="Eliminates a useless feature that wastes model capacity.",
            ))
            drop_cands.append(col)

        # ── TEXT → flag for NLP ───────────────────────────────────────────
        elif profile.column_type == ColumnType.TEXT:
            recs.append(FeatureRecommendation(
                column=col, action="extract",
                reason=f"'{col}' contains free-form text. Standard encoding is not appropriate.",
                priority="medium",
                impact="TF-IDF or embedding can unlock significant predictive power from text.",
            ))

        # ── CATEGORICAL — check cardinality ──────────────────────────────
        elif profile.column_type == ColumnType.CATEGORICAL:
            if profile.n_unique > 50:
                recs.append(FeatureRecommendation(
                    column=col, action="encode",
                    reason=f"'{col}' has {profile.n_unique} unique values — high cardinality. OHE will create {profile.n_unique} columns.",
                    priority="high",
                    impact="Use Frequency or Target Encoding to avoid dimensionality explosion.",
                ))
                encode_cands.append(col)
            else:
                encode_cands.append(col)

        # ── NUMERICAL — check skewness ────────────────────────────────────
        elif profile.column_type == ColumnType.NUMERICAL:
            series = df[col].dropna()
            if len(series) > 10:
                skew = float(series.skew())
                if abs(skew) > 1.5 and series.min() > 0:
                    recs.append(FeatureRecommendation(
                        column=col, action="log_transform",
                        reason=f"'{col}' has high skewness ({skew:.2f}) and positive values. Log transform may improve model fit.",
                        priority="medium",
                        impact="Reduces skew, making linear models and distance-based algorithms more effective.",
                    ))
                    transform_cands.append(col)

        # ── DATETIME → extract ────────────────────────────────────────────
        elif profile.column_type == ColumnType.DATETIME:
            recs.append(FeatureRecommendation(
                column=col, action="extract",
                reason=f"'{col}' is a datetime. Extract year, month, dayofweek, etc.",
                priority="medium",
                impact="Temporal features often carry strong seasonal and trend signals.",
            ))

    n_high = sum(1 for r in recs if r.priority == "high")
    if n_high > 0:
        summary = f"{n_high} high-priority recommendations. Address these before training."
    elif recs:
        summary = f"{len(recs)} feature recommendations available."
    else:
        summary = "Features look well-structured. No major preprocessing changes needed."

    return FeatureRecommendationReport(
        recommendations=recs,
        drop_candidates=drop_cands,
        encode_candidates=encode_cands,
        transform_candidates=transform_cands,
        summary=summary,
    )
