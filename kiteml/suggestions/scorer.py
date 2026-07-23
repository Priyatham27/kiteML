"""
scorer.py — Confidence scoring and ranking utilities for KiteML suggestions.
"""

from typing import Sequence

from kiteml.suggestions.result import Suggestion


def rank_suggestions(
    suggestions: Sequence[Suggestion],
    min_confidence: float = 0.5,
    top_k: int = 5,
) -> list[Suggestion]:
    """
    Filter and rank suggestions by confidence score.

    Parameters
    ----------
    suggestions : Sequence of Suggestion
        Input list of generated suggestions.
    min_confidence : float
        Minimum confidence threshold (0.0 to 1.0).
    top_k : int
        Maximum number of top suggestions to return.

    Returns
    -------
    list of Suggestion
        Filtered and sorted list of top suggestions.
    """
    filtered = [s for s in suggestions if s.confidence >= min_confidence]
    filtered.sort(key=lambda s: s.confidence, reverse=True)
    return filtered[:top_k]
