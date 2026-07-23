"""
suggestion_manager.py — SuggestionManager gateway for generating and rendering recommendations.
"""

from typing import Any

from kiteml.suggestions import (
    Suggestion,
    SuggestionEngine,
    SuggestionFormatter,
)


class SuggestionManager:
    """
    Modular manager gateway for generating and formatting context-aware suggestions.
    """

    def __init__(
        self,
        engine: SuggestionEngine | None = None,
        formatter: SuggestionFormatter | None = None,
    ) -> None:
        self.engine = engine or SuggestionEngine()
        self.formatter = formatter or SuggestionFormatter()

    def generate(
        self,
        source: Any,
        min_confidence: float | None = None,
        top_k: int | None = None,
    ) -> list[Suggestion]:
        """Generate ranked suggestions for any error, warning, dictionary, or DataFrame context."""
        return self.engine.generate(source, min_confidence=min_confidence, top_k=top_k)

    def format_suggestions(
        self,
        suggestions: list[Suggestion],
        mode: str = "terminal",
    ) -> str:
        """Format suggestions list into terminal, text, or JSON mode."""
        return self.formatter.format(suggestions, mode=mode)
