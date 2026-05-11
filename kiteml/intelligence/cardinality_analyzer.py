"""
cardinality_analyzer.py — Cardinality analysis for categorical columns.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class CardinalityInfo:
    column: str
    n_unique: int
    cardinality_level: str  # "low" | "medium" | "high" | "very_high"
    rare_categories: list[Any]
    rare_count: int
    top_value: Any
    top_freq: float
    encoding_recommendation: str


@dataclass
class CardinalityReport:
    columns_analyzed: int
    high_cardinality_columns: list[str]
    details: dict[str, CardinalityInfo]
    recommendations: list[str]


def analyze_cardinality(
    df: pd.DataFrame,
    low_threshold: int = 10,
    medium_threshold: int = 50,
    high_threshold: int = 200,
    rare_freq_threshold: float = 0.01,
) -> CardinalityReport:
    """Analyze cardinality of categorical/object columns."""
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    details: dict[str, CardinalityInfo] = {}
    high_card: list[str] = []
    recommendations: list[str] = []

    for col in cat_cols:
        series = df[col].dropna()
        n_unique = int(series.nunique())

        if n_unique <= low_threshold:
            level, enc_rec = "low", "One-Hot Encoding recommended."
        elif n_unique <= medium_threshold:
            level, enc_rec = "medium", "OHE acceptable; consider Target Encoding for trees."
        elif n_unique <= high_threshold:
            level, enc_rec = "high", "Use Frequency or Target Encoding instead of OHE."
        else:
            level = "very_high"
            enc_rec = f"⚠️ {n_unique} unique values. Use Frequency/Hash encoding or drop."
            high_card.append(col)

        vc = series.value_counts(normalize=True)
        rare = [val for val, freq in vc.items() if freq < rare_freq_threshold]
        top_val = vc.index[0] if len(vc) > 0 else None
        top_freq = round(float(vc.iloc[0]), 4) if len(vc) > 0 else 0.0

        details[col] = CardinalityInfo(
            column=col,
            n_unique=n_unique,
            cardinality_level=level,
            rare_categories=rare[:10],
            rare_count=len(rare),
            top_value=top_val,
            top_freq=top_freq,
            encoding_recommendation=enc_rec,
        )

    if high_card:
        recommendations.append(f"High-cardinality columns: {high_card}. Avoid OHE.")

    return CardinalityReport(
        columns_analyzed=len(cat_cols),
        high_cardinality_columns=high_card,
        details=details,
        recommendations=recommendations,
    )
