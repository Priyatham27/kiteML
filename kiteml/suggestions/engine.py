"""
engine.py — SuggestionEngine orchestrator for KiteML.
"""

from typing import Any, Sequence

from kiteml.suggestions.context import SuggestionContext
from kiteml.suggestions.providers import (
    BaseSuggestionProvider,
    ColumnSuggestionProvider,
    DeploymentSuggestionProvider,
    PerformanceSuggestionProvider,
    SchemaSuggestionProvider,
    TargetSuggestionProvider,
    TrainingSuggestionProvider,
    ValidationSuggestionProvider,
)
from kiteml.suggestions.result import Suggestion
from kiteml.suggestions.scorer import rank_suggestions


class SuggestionEngine:
    """
    Central engine orchestrating suggestion providers to generate ranked recommendations.
    """

    def __init__(
        self,
        providers: Sequence[BaseSuggestionProvider] | None = None,
        min_confidence: float = 0.5,
        top_k: int = 5,
    ) -> None:
        self.min_confidence = min_confidence
        self.top_k = top_k
        self.providers: list[BaseSuggestionProvider] = (
            list(providers) if providers is not None else self._default_providers()
        )

    def _default_providers(self) -> list[BaseSuggestionProvider]:
        """Return default suite of suggestion providers."""
        return [
            ColumnSuggestionProvider(),
            TargetSuggestionProvider(),
            SchemaSuggestionProvider(),
            ValidationSuggestionProvider(),
            TrainingSuggestionProvider(),
            DeploymentSuggestionProvider(),
            PerformanceSuggestionProvider(),
        ]

    def register_provider(self, provider: BaseSuggestionProvider) -> None:
        """Register a custom suggestion provider."""
        self.providers.append(provider)

    def generate(
        self,
        source: Any,
        min_confidence: float | None = None,
        top_k: int | None = None,
    ) -> list[Suggestion]:
        """
        Generate ranked suggestions from context, exception, dictionary, or DataFrame.

        Parameters
        ----------
        source : Any
            Input context, KiteMLError, dictionary, or DataFrame.
        min_confidence : float, optional
            Minimum confidence threshold (defaults to engine min_confidence).
        top_k : int, optional
            Maximum suggestions to return (defaults to engine top_k).

        Returns
        -------
        list of Suggestion
            Sorted list of top ranked suggestions.
        """
        context = SuggestionContext.from_input(source)
        all_suggestions: list[Suggestion] = []

        for provider in self.providers:
            try:
                res = provider.generate(context)
                all_suggestions.extend(res)
            except Exception:
                # Suggestion generation should never crash execution
                continue

        thresh = min_confidence if min_confidence is not None else self.min_confidence
        limit = top_k if top_k is not None else self.top_k

        return rank_suggestions(all_suggestions, min_confidence=thresh, top_k=limit)
