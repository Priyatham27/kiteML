"""
error_manager.py — ErrorManager orchestrator for KiteML exceptions.
"""

from typing import Any

from kiteml.exceptions import ErrorCatalog, ErrorFormatter, KiteMLError, wrap_exception
from kiteml.pipeline.context_builder import ContextBuilder
from kiteml.pipeline.suggestion_manager import SuggestionManager


class ErrorManager:
    """
    Central orchestration layer for exceptions.
    Catches errors, enriches context, fetches catalog metadata, generates suggestions,
    and returns formatted output.
    """

    def __init__(
        self,
        suggestion_manager: SuggestionManager | None = None,
        context_builder: ContextBuilder | None = None,
        formatter: ErrorFormatter | None = None,
    ) -> None:
        self.suggestion_manager = suggestion_manager or SuggestionManager()
        self.context_builder = context_builder or ContextBuilder()
        self.formatter = formatter or ErrorFormatter()

    def process_error(
        self,
        error: Exception,
        extra_context: dict[str, Any] | None = None,
    ) -> KiteMLError:
        """
        Process any exception into a fully enriched KiteMLError.
        """
        kml_err = wrap_exception(error) if not isinstance(error, KiteMLError) else error

        if extra_context and hasattr(kml_err, "context"):
            if hasattr(kml_err.context, "extra"):
                kml_err.context.extra.update(extra_context)

        # Lookup catalog definition if code present
        if kml_err.error_code:
            defn = ErrorCatalog.get(kml_err.error_code)
            if defn and not kml_err.suggestion:
                kml_err.suggestion = defn.default_suggestion

        return kml_err

    def format_error(
        self,
        error: Exception,
        mode: str = "terminal",
        extra_context: dict[str, Any] | None = None,
    ) -> str:
        """
        Enrich and format an exception into a developer-friendly output string.
        """
        kml_err = self.process_error(error, extra_context=extra_context)
        return self.formatter.format(kml_err, mode=mode)
