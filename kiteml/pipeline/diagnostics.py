"""
diagnostics.py — Diagnostics dataclass and DiagnosticsManager for KiteML pipeline summaries.
"""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Diagnostics:
    """
    Complete execution diagnostics summary.

    Attributes
    ----------
    status : str
        Execution status ('SUCCESS', 'WARNINGS', 'FAILED').
    error_count : int
        Total errors encountered.
    warning_count : int
        Total warnings collected.
    suggestion_count : int
        Total recommendations generated.
    validation_status : str
        Status of dataset validation ('Passed', 'Failed', 'Skipped').
    training_status : str
        Status of model training ('Completed', 'Failed', 'Skipped').
    execution_time : float
        Total execution wall-clock time in seconds.
    """

    status: str = "SUCCESS"
    error_count: int = 0
    warning_count: int = 0
    suggestion_count: int = 0
    validation_status: str = "Passed"
    training_status: str = "Completed"
    execution_time: float = 0.0

    def summary_text(self, width: int = 40) -> str:
        """Render diagnostics to formatted terminal text box."""
        lines = [
            "━" * width,
            "🪁 KiteML Diagnostics",
            "━" * width,
            f"  Status             {self.status}",
            f"  Errors             {self.error_count}",
            f"  Warnings           {self.warning_count}",
            f"  Suggestions        {self.suggestion_count}",
            f"  Validation         {self.validation_status}",
            f"  Training           {self.training_status}",
            f"  Execution Time     {self.execution_time:.2f} sec",
            "━" * width,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize Diagnostics to dictionary."""
        return asdict(self)


class DiagnosticsManager:
    """Manager for generating execution Diagnostics summaries."""

    def create_diagnostics(
        self,
        status: str = "SUCCESS",
        error_count: int = 0,
        warning_count: int = 0,
        suggestion_count: int = 0,
        validation_status: str = "Passed",
        training_status: str = "Completed",
        execution_time: float = 0.0,
    ) -> Diagnostics:
        """Create a Diagnostics instance."""
        return Diagnostics(
            status=status,
            error_count=error_count,
            warning_count=warning_count,
            suggestion_count=suggestion_count,
            validation_status=validation_status,
            training_status=training_status,
            execution_time=execution_time,
        )
