"""
correlation_analyzer.py — Multicollinearity and feature correlation analysis.

Detects:
  - Highly correlated feature pairs (multicollinearity risk)
  - Strongest predictors of the target
  - Redundant features that could be safely dropped
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class CorrelationPair:
    """A pair of correlated features."""

    col_a: str
    col_b: str
    correlation: float
    is_redundant: bool  # True if |corr| > high_threshold


@dataclass
class CorrelationReport:
    """Full correlation analysis of a DataFrame."""

    high_correlation_pairs: list[CorrelationPair]
    target_correlations: dict[str, float]  # col → |corr| with target
    top_predictors: list[str]  # sorted by |corr| with target
    redundant_features: list[str]  # likely safe to drop
    recommendations: list[str]


def analyze_correlations(
    df: pd.DataFrame,
    target: Optional[str] = None,
    high_threshold: float = 0.90,
    moderate_threshold: float = 0.70,
) -> CorrelationReport:
    """
    Analyze feature correlations and detect multicollinearity.

    Parameters
    ----------
    df : pd.DataFrame
    target : str, optional
        Target column name for target-correlation analysis.
    high_threshold : float
        |Pearson r| above which features are considered redundant. Default 0.90.
    moderate_threshold : float
        |Pearson r| above which pairs are flagged. Default 0.70.

    Returns
    -------
    CorrelationReport
    """
    numeric_df = df.select_dtypes(include=[np.number])
    feature_cols = [c for c in numeric_df.columns if c != target]

    high_pairs: list[CorrelationPair] = []
    redundant: list[str] = []
    recommendations: list[str] = []

    # ── Feature-feature correlations ─────────────────────────────────────
    if len(feature_cols) >= 2:
        corr_matrix = numeric_df[feature_cols].corr().abs()
        # Upper triangle only
        seen_redundant = set()
        for i, col_a in enumerate(feature_cols):
            for j, col_b in enumerate(feature_cols):
                if j <= i:
                    continue
                r = corr_matrix.loc[col_a, col_b]
                if r >= moderate_threshold:
                    is_redundant = r >= high_threshold
                    high_pairs.append(
                        CorrelationPair(
                            col_a=col_a,
                            col_b=col_b,
                            correlation=round(float(r), 4),
                            is_redundant=is_redundant,
                        )
                    )
                    if is_redundant and col_b not in seen_redundant:
                        redundant.append(col_b)
                        seen_redundant.add(col_b)

    if high_pairs:
        recommendations.append(f"{len(high_pairs)} highly correlated feature pair(s) detected.")
    if redundant:
        recommendations.append(
            f"Consider dropping: {redundant[:5]} (corr ≥ {high_threshold:.0%} with another feature)."
        )

    # ── Target correlations ───────────────────────────────────────────────
    target_corrs: dict[str, float] = {}
    top_predictors: list[str] = []

    if target and target in numeric_df.columns:
        corrs = numeric_df[feature_cols].corrwith(numeric_df[target]).abs().dropna()
        target_corrs = {col: round(float(v), 4) for col, v in corrs.items()}
        top_predictors = sorted(target_corrs, key=lambda c: target_corrs[c], reverse=True)[:10]

        if top_predictors:
            top_col = top_predictors[0]
            recommendations.append(f"Strongest predictor: '{top_col}' (|r|={target_corrs[top_col]:.3f})")

    return CorrelationReport(
        high_correlation_pairs=high_pairs,
        target_correlations=target_corrs,
        top_predictors=top_predictors,
        redundant_features=redundant,
        recommendations=recommendations,
    )
