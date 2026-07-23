"""
context.py — SuggestionContext dataclass for storing execution state for suggestions.
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class SuggestionContext:
    """
    Context model passed to SuggestionProviders.
    """

    df: Any | None = None
    target: str | None = None
    problem_type: str | None = None
    error: Any | None = None
    warning: Any | None = None
    available_columns: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.available_columns and isinstance(self.df, pd.DataFrame):
            self.available_columns = list(self.df.columns)

    @classmethod
    def from_input(cls, source: Any) -> "SuggestionContext":
        """Build SuggestionContext from an exception, dictionary, or DataFrame."""
        if isinstance(source, cls):
            return source
        if isinstance(source, dict):
            return cls(**{k: v for k, v in source.items() if k in cls.__dataclass_fields__})
        if isinstance(source, pd.DataFrame):
            return cls(df=source, available_columns=list(source.columns))
        # If source is an exception (e.g. KiteMLError)
        error_ctx = getattr(source, "context", None)
        cols: list[Any] = []
        target = getattr(source, "target", None)
        ctx_dict = {}
        if isinstance(error_ctx, dict):
            ctx_dict = error_ctx
            cols = error_ctx.get("available_columns") or error_ctx.get("columns") or []
            target = target or error_ctx.get("target")
        elif error_ctx is not None and hasattr(error_ctx, "to_dict"):
            ctx_dict = error_ctx.to_dict()
            cols = ctx_dict.get("available_columns") or ctx_dict.get("columns") or []
            target = target or ctx_dict.get("target")

        return cls(
            error=source,
            target=target,
            available_columns=list(cols) if cols else [],
            metadata=ctx_dict,
        )
