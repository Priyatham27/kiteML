"""
test_providers.py — Unit tests for specialized suggestion providers (Story 3.5).
"""

import pandas as pd
import pytest

from kiteml.suggestions.context import SuggestionContext
from kiteml.suggestions.providers import (
    ColumnSuggestionProvider,
    SchemaSuggestionProvider,
    TargetSuggestionProvider,
    ValidationSuggestionProvider,
)


def test_column_suggestion_provider():
    provider = ColumnSuggestionProvider()
    ctx = SuggestionContext(target="prcie", available_columns=["Price", "Age", "City"])

    suggestions = provider.generate(ctx)
    assert len(suggestions) > 0
    assert suggestions[0].title == "Did you mean 'Price'?"
    assert len(suggestions[0].why) > 0


def test_target_suggestion_provider():
    provider = TargetSuggestionProvider()
    df = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "label": [0, 1, 0, 1, 0],
        }
    )
    ctx = SuggestionContext(df=df)

    suggestions = provider.generate(ctx)
    assert len(suggestions) > 0
    assert "label" in suggestions[0].title


def test_schema_suggestion_provider():
    provider = SchemaSuggestionProvider()
    df = pd.DataFrame(
        {
            "const_col": [1, 1, 1, 1],
            "norm_col": [1, 2, 3, 4],
        }
    )
    ctx = SuggestionContext(df=df)

    suggestions = provider.generate(ctx)
    assert len(suggestions) == 1
    assert "const_col" in suggestions[0].title


def test_validation_suggestion_provider():
    provider = ValidationSuggestionProvider()
    df = pd.DataFrame(
        {
            "missing_col": [1.0, 2.0, None, 4.0, 5.0],
            "clean_col": [1, 2, 3, 4, 5],
        }
    )
    ctx = SuggestionContext(df=df)

    suggestions = provider.generate(ctx)
    assert len(suggestions) == 1
    assert "Median Imputation" in suggestions[0].title
