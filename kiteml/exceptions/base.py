"""
base.py — KiteMLError root exception for KiteML.
"""

import json
import time
from typing import Any

from kiteml.exceptions.codes import KML_ERR_GENERIC
from kiteml.exceptions.context import ErrorContext


class KiteMLError(Exception):
    """
    Root exception for all KiteML errors.

    Carries structured metadata including error codes, suggestions,
    ErrorContext models, details, and serialization utilities.
    """

    def __init__(
        self,
        message: str,
        error_code: str = KML_ERR_GENERIC,
        severity: str = "error",
        suggestion: str | None = None,
        context: ErrorContext | dict[str, Any] | None = None,
        details: str | None = None,
        timestamp: float | None = None,
        help_url: str | None = None,
    ) -> None:
        from kiteml.exceptions.catalog import ErrorCatalog

        defn = ErrorCatalog.get(error_code) if error_code else None

        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity or (defn.severity if defn else "error")
        self.suggestion = suggestion or (defn.default_suggestion if defn else None)
        self.details = details
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.help_url = help_url or (
            f"https://kiteml.org/docs/errors/{defn.documentation_slug}" if defn and defn.documentation_slug else None
        )

        if isinstance(context, ErrorContext):
            self.context = context
        elif isinstance(context, dict):
            self.context = ErrorContext.from_dict(context)
        else:
            self.context = ErrorContext()

    def to_dict(self) -> dict[str, Any]:
        """Convert exception and metadata into a dictionary."""
        result: dict[str, Any] = {
            "error_class": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "severity": self.severity,
            "timestamp": self.timestamp,
        }
        if self.suggestion:
            result["suggestion"] = self.suggestion
        if self.details:
            result["details"] = self.details
        if self.help_url:
            result["help_url"] = self.help_url

        ctx_dict = self.context.to_dict()
        if ctx_dict:
            result["context"] = ctx_dict

        if self.__cause__ is not None:
            result["cause"] = f"{type(self.__cause__).__name__}: {self.__cause__}"
        elif self.__context__ is not None and not self.__suppress_context__:
            result["cause"] = f"{type(self.__context__).__name__}: {self.__context__}"

        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize exception to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def suggestions(self) -> list[Any]:
        """Generate context-aware ranked recommendations for this exception."""
        from kiteml.suggestions.engine import SuggestionEngine

        engine = SuggestionEngine()
        return engine.generate(self)

    def __str__(self) -> str:
        code_str = f"[{self.error_code}] " if self.error_code else ""
        msg = f"{code_str}{self.message}"
        if self.suggestion:
            msg += f"\n💡 Suggestion: {self.suggestion}"
        return msg

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} [{self.error_code}]: {self.message}>"
