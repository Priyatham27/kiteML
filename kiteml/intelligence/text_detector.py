"""
text_detector.py — Detect NLP-candidate columns in a DataFrame.

A column is a text candidate when it contains free-form natural language:
long strings, high vocabulary diversity, many unique values.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd


@dataclass
class TextColumnInfo:
    """Metadata about a detected text column."""
    column: str
    avg_word_count: float
    avg_char_count: float
    unique_ratio: float
    confidence: float
    suggestion: str


@dataclass
class TextDetectionResult:
    """Result of text column detection across a DataFrame."""
    text_columns: List[str]
    details: Dict[str, TextColumnInfo]
    has_text: bool


def detect_text_columns(
    df: pd.DataFrame,
    min_avg_words: float = 4.0,
    min_unique_ratio: float = 0.3,
) -> TextDetectionResult:
    """
    Detect columns that contain free-form natural language text.

    Parameters
    ----------
    df : pd.DataFrame
    min_avg_words : float
        Minimum average word count to flag as text. Default 4.
    min_unique_ratio : float
        Minimum unique value ratio below which a column is too repetitive to
        be considered free text.

    Returns
    -------
    TextDetectionResult
    """
    text_cols: List[str] = []
    details: Dict[str, TextColumnInfo] = {}

    string_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in string_cols:
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            continue

        sample = series.head(200)
        avg_words = float(sample.str.split().str.len().mean())
        avg_chars = float(sample.str.len().mean())
        unique_ratio = float(series.nunique() / len(series))

        # Scoring
        confidence = 0.0
        if avg_words >= min_avg_words:
            confidence += 0.5
        if avg_words >= 8:
            confidence += 0.2
        if unique_ratio >= min_unique_ratio:
            confidence += 0.2
        if avg_chars > 50:
            confidence += 0.1

        if confidence >= 0.5:
            suggestion = (
                "Consider TF-IDF vectorization or sentence embeddings for ML."
                if avg_words > 6 else
                "Consider bag-of-words or label encoding."
            )
            info = TextColumnInfo(
                column=col,
                avg_word_count=round(avg_words, 2),
                avg_char_count=round(avg_chars, 2),
                unique_ratio=round(unique_ratio, 4),
                confidence=round(confidence, 2),
                suggestion=suggestion,
            )
            text_cols.append(col)
            details[col] = info

    return TextDetectionResult(
        text_columns=text_cols,
        details=details,
        has_text=len(text_cols) > 0,
    )
