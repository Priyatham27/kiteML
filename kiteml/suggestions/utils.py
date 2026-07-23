"""
utils.py — Helper functions for the KiteML suggestions engine.
"""

from typing import Any

from kiteml.suggestions.engine import SuggestionEngine
from kiteml.suggestions.result import Suggestion


def generate_suggestions(source: Any, min_confidence: float = 0.5, top_k: int = 5) -> list[Suggestion]:
    """Shortcut function to generate top ranked suggestions from any input source."""
    engine = SuggestionEngine(min_confidence=min_confidence, top_k=top_k)
    return engine.generate(source)
