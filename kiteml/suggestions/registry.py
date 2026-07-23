"""
registry.py — SuggestionRegistry container for KiteML suggestions.
"""

from typing import Sequence

from kiteml.suggestions.engine import SuggestionEngine
from kiteml.suggestions.providers import BaseSuggestionProvider


class SuggestionRegistry:
    """Global registry holding shared SuggestionEngine instance."""

    def __init__(self) -> None:
        self.engine = SuggestionEngine()

    def register(self, provider: BaseSuggestionProvider) -> None:
        """Register a new provider globally."""
        self.engine.register_provider(provider)

    def providers(self) -> Sequence[BaseSuggestionProvider]:
        """Return registered providers."""
        return self.engine.providers


global_suggestion_registry = SuggestionRegistry()
