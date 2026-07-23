"""
dx_pipeline.py — DXPipeline central Developer Experience orchestrator.
"""

import time
from typing import Any

from kiteml.exceptions import KiteMLError
from kiteml.pipeline.context_builder import ContextBuilder
from kiteml.pipeline.diagnostics import Diagnostics, DiagnosticsManager
from kiteml.pipeline.error_manager import ErrorManager
from kiteml.pipeline.suggestion_manager import SuggestionManager
from kiteml.pipeline.warning_manager import WarningManager
from kiteml.suggestions.result import Suggestion
from kiteml.warnings.base import KiteMLWarning


class DXPipeline:
    """
    Developer Experience Pipeline (DX Pipeline).
    Centralized architectural boundary coordinating context, errors, warnings,
    suggestions, formatting, and diagnostics across KiteML workflows.
    """

    def __init__(
        self,
        context_builder: ContextBuilder | None = None,
        error_manager: ErrorManager | None = None,
        warning_manager: WarningManager | None = None,
        suggestion_manager: SuggestionManager | None = None,
        diagnostics_manager: DiagnosticsManager | None = None,
    ) -> None:
        self.context_builder = context_builder or ContextBuilder()
        self.suggestion_manager = suggestion_manager or SuggestionManager()
        self.error_manager = error_manager or ErrorManager(
            suggestion_manager=self.suggestion_manager,
            context_builder=self.context_builder,
        )
        self.warning_manager = warning_manager or WarningManager()
        self.diagnostics_manager = diagnostics_manager or DiagnosticsManager()

        self._start_time: float = 0.0
        self._error_count: int = 0
        self._validation_status: str = "Passed"
        self._training_status: str = "Not Started"

    def start(self) -> None:
        """Start tracking pipeline execution time."""
        self._start_time = time.time()
        self._error_count = 0
        self._validation_status = "Passed"
        self._training_status = "Not Started"

    def add_warning(self, warning: KiteMLWarning) -> None:
        """Add a warning to the warning manager."""
        self.warning_manager.add(warning)

    def process_error(self, error: Exception, extra_context: dict[str, Any] | None = None) -> KiteMLError:
        """Process and enrich an exception through ErrorManager."""
        self._error_count += 1
        return self.error_manager.process_error(error, extra_context=extra_context)

    def set_validation_status(self, status: str) -> None:
        """Update validation status ('Passed', 'Failed', 'Skipped')."""
        self._validation_status = status

    def set_training_status(self, status: str) -> None:
        """Update training status ('Completed', 'Failed', 'Skipped')."""
        self._training_status = status

    def get_suggestions(self, source: Any) -> list[Suggestion]:
        """Generate suggestions for any execution context."""
        return self.suggestion_manager.generate(source)

    def get_diagnostics(self) -> Diagnostics:
        """Calculate and return execution diagnostics."""
        elapsed = time.time() - self._start_time if self._start_time > 0 else 0.0
        status = (
            "FAILED" if self._error_count > 0 else ("WARNINGS" if len(self.warning_manager.warnings) > 0 else "SUCCESS")
        )

        return self.diagnostics_manager.create_diagnostics(
            status=status,
            error_count=self._error_count,
            warning_count=len(self.warning_manager.warnings),
            suggestion_count=len(self.get_suggestions(self.warning_manager.warnings)),
            validation_status=self._validation_status,
            training_status=self._training_status,
            execution_time=elapsed,
        )
