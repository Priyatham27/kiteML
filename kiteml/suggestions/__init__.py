"""
KiteML Context-Aware Suggestions Engine Package.

Intelligent recommendations with explainable reasoning.
"""

from kiteml.suggestions.context import SuggestionContext
from kiteml.suggestions.engine import SuggestionEngine
from kiteml.suggestions.formatter import SuggestionFormatter
from kiteml.suggestions.matcher import levenshtein_distance, match_column_name, string_similarity
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
from kiteml.suggestions.registry import (
    SuggestionRegistry,
    global_suggestion_registry,
)
from kiteml.suggestions.result import Suggestion
from kiteml.suggestions.scorer import rank_suggestions
from kiteml.suggestions.utils import generate_suggestions

__all__ = [
    "Suggestion",
    "SuggestionContext",
    "SuggestionEngine",
    "SuggestionFormatter",
    "BaseSuggestionProvider",
    "ColumnSuggestionProvider",
    "TargetSuggestionProvider",
    "SchemaSuggestionProvider",
    "ValidationSuggestionProvider",
    "TrainingSuggestionProvider",
    "DeploymentSuggestionProvider",
    "PerformanceSuggestionProvider",
    "SuggestionRegistry",
    "global_suggestion_registry",
    "generate_suggestions",
    "levenshtein_distance",
    "string_similarity",
    "match_column_name",
    "rank_suggestions",
]
