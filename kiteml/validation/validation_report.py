"""
validation_report.py — Aggregate report generator for KiteML dataset validation.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from kiteml.validation.utils import format_bytes, format_number
from kiteml.validation.validation_result import ValidationResult


@dataclass
class ValidationReport:
    """
    Aggregated validation report combining results from multiple validators.

    Attributes
    ----------
    results : list[ValidationResult]
        List of ValidationResult objects produced by executed validators.
    dataset_metadata : dict[str, Any]
        General metadata about the dataset (e.g. n_rows, n_cols, memory_bytes).
    total_execution_time : float
        Combined execution time in seconds across all validators.
    """

    results: list[ValidationResult] = field(default_factory=list)
    dataset_metadata: dict[str, Any] = field(default_factory=dict)
    total_execution_time: float = 0.0

    @property
    def passed(self) -> bool:
        """True if all validator results passed without errors or critical issues."""
        return not self.has_errors()

    def has_errors(self) -> bool:
        """True if any result contains ERROR or CRITICAL messages."""
        return any(r.has_errors() for r in self.results)

    def has_warnings(self) -> bool:
        """True if any result contains WARNING messages."""
        return any(r.has_warnings() for r in self.results)

    def add_result(self, result: ValidationResult) -> None:
        """Add a ValidationResult to the report."""
        self.results.append(result)

    @property
    def total_messages_count(self) -> int:
        """Total number of validation messages recorded."""
        return sum(len(r.messages) for r in self.results)

    @property
    def summary(self) -> dict[str, int]:
        """Aggregate summary of checks, warnings, and errors."""
        total_errors = sum(len(r.errors) for r in self.results)
        total_warnings = sum(len(r.warnings) for r in self.results)
        total_infos = sum(len(r.infos) for r in self.results)
        total_checks = sum(len(r.messages) for r in self.results)
        # Total rules/checks that passed without issues
        total_passed_checks = max(0, total_checks - total_errors - total_warnings)

        return {
            "checks": total_checks,
            "passed": total_passed_checks,
            "warnings": total_warnings,
            "errors": total_errors,
            "infos": total_infos,
        }

    def render_terminal(self) -> str:
        """
        Generate a formatted terminal text report.

        Returns
        -------
        str
        """
        lines = []
        lines.append("══════════════════════════════")
        lines.append("🪁 KiteML Dataset Validation")
        lines.append("══════════════════════════════")
        lines.append("")

        # Dataset Section
        lines.append("Dataset")
        lines.append("──────────────")
        n_rows = self.dataset_metadata.get("n_rows")
        n_cols = self.dataset_metadata.get("n_cols")
        mem_mb = self.dataset_metadata.get("memory_mb")
        mem_bytes = self.dataset_metadata.get("memory_bytes")

        if n_rows is not None:
            lines.append(f"Rows: {format_number(n_rows)}")
        if n_cols is not None:
            lines.append(f"Columns: {format_number(n_cols)}")
        if mem_mb is not None:
            lines.append(f"Memory: {mem_mb:.1f} MB")
        elif mem_bytes is not None:
            lines.append(f"Memory: {format_bytes(mem_bytes)}")
        lines.append("")

        # Checks Section
        lines.append("Checks")
        lines.append("──────────────")
        has_any_check = False
        for r in self.results:
            if not r.messages:
                lines.append(f"✓ {r.validator_name} Passed")
                has_any_check = True
                continue

            for msg in r.messages:
                has_any_check = True
                if msg.severity.value in ("error", "critical"):
                    symbol = "❌"
                elif msg.severity.value == "warning":
                    symbol = "⚠"
                else:
                    symbol = "✓"

                lines.append(f"{symbol} {msg.title}")
                if msg.description:
                    lines.append(f"  {msg.description}")

        if not has_any_check:
            lines.append("✓ No issues detected.")

        lines.append("")

        # Summary Section
        sum_stats = self.summary
        lines.append("Summary")
        lines.append("──────────────")
        lines.append(f"Passed: {sum_stats['passed']}")
        lines.append(f"Warnings: {sum_stats['warnings']}")
        lines.append(f"Errors: {sum_stats['errors']}")
        lines.append("")

        if self.passed:
            lines.append("Ready for Training")
        else:
            lines.append("Validation Failed — Fix errors before training")
        lines.append("══════════════════════════════")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert aggregate report to a structured dictionary."""
        return {
            "passed": self.passed,
            "summary": self.summary,
            "total_execution_time": self.total_execution_time,
            "dataset_metadata": self.dataset_metadata,
            "validators": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return f"<ValidationReport(status='{status}', validators={len(self.results)})>"
