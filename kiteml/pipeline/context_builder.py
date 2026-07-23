"""
context_builder.py — ContextBuilder for creating standardized execution context in KiteML.
"""

from typing import Any

import pandas as pd

from kiteml.suggestions.context import SuggestionContext


class ContextBuilder:
    """
    Constructs standardized execution context for errors, warnings, and suggestions.
    """

    def build_context(
        self,
        df: Any | None = None,
        target: str | None = None,
        pipeline_stage: str = "general",
        model_name: str | None = None,
        execution_time: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Build standardized context dictionary.
        """
        rows = len(df) if isinstance(df, pd.DataFrame) else 0
        cols = list(df.columns) if isinstance(df, pd.DataFrame) else []

        context: dict[str, Any] = {
            "pipeline_stage": pipeline_stage,
            "dataset_rows": rows,
            "dataset_columns": len(cols),
            "available_columns": cols,
            "target": target,
            "model_name": model_name,
            "execution_time": execution_time,
        }

        if extra:
            context.update(extra)

        return context

    def to_suggestion_context(
        self,
        df: Any | None = None,
        target: str | None = None,
        error: Any | None = None,
        warning: Any | None = None,
        problem_type: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> SuggestionContext:
        """
        Convert execution state into a SuggestionContext instance.
        """
        cols = list(df.columns) if isinstance(df, pd.DataFrame) else []
        return SuggestionContext(
            df=df,
            target=target,
            problem_type=problem_type,
            error=error,
            warning=warning,
            available_columns=cols,
            metadata=extra or {},
        )
