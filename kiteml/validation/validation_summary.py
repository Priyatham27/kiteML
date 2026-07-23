"""
validation_summary.py — Consolidated ValidationSummary representation for KiteML.
"""

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationSummary:
    """
    Consolidated validation summary object aggregating all pipeline validation results.

    Attributes
    ----------
    passed : bool
        True if all critical and error checks passed.
    health_score : int
        Overall dataset health score (0–100).
    health_grade : str
        Letter grade ('A+', 'A', 'B', 'C', 'Needs Attention').
    health_rating : str
        Star rating display string ('★★★★★ Excellent', etc.).
    total_checks : int
        Total number of validation rules executed.
    passed_checks : int
        Total number of rules passed without warnings/errors.
    warning_count : int
        Total number of warning messages.
    error_count : int
        Total number of error messages.
    critical_count : int
        Total number of critical messages.
    execution_time : float
        Total pipeline execution time in seconds.
    validator_results : dict[str, Any]
        Dictionary mapping validator names to their individual ValidationResult dictionaries.
    recommendations : list[str]
        Actionable data quality recommendations.
    ready_for_training : bool
        True if dataset is safe for machine learning training.
    """

    passed: bool = True
    health_score: int = 100
    health_grade: str = "A+"
    health_rating: str = "★★★★★ Excellent"
    total_checks: int = 0
    passed_checks: int = 0
    warning_count: int = 0
    error_count: int = 0
    critical_count: int = 0
    execution_time: float = 0.0
    validator_results: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    ready_for_training: bool = True

    def summary_text(self) -> str:
        """Render human-readable formatted terminal summary."""
        lines = [
            "══════════════════════════════════════════════",
            "🪁 KiteML Validation Report",
            "══════════════════════════════════════════════",
        ]

        # Dataset Metrics from DatasetValidator / QualityValidator
        ds_stats = self.validator_results.get("DatasetValidator", {}).get("statistics", {})
        n_rows = ds_stats.get("n_rows", "N/A")
        n_cols = ds_stats.get("n_cols", "N/A")
        memory_mb = ds_stats.get("memory_mb", "N/A")

        lines.extend(
            [
                f"Rows            {n_rows}",
                f"Columns         {n_cols}",
                (
                    f"Memory          {memory_mb} MB"
                    if isinstance(memory_mb, (int, float))
                    else f"Memory          {memory_mb}"
                ),
                "--------------------------------------------",
            ]
        )

        # Per Validator Summaries
        for val_name, val_data in self.validator_results.items():
            val_passed = val_data.get("passed", True)
            messages = val_data.get("messages", [])
            status_icon = "✓ Passed" if val_passed else "❌ Failed"
            lines.append(f"{val_name}")
            lines.append(f"{status_icon}")

            # Highlight Warnings or Errors
            for msg in messages:
                sev = msg.get("severity", "").lower()
                title = msg.get("title", "")
                if sev in ("warning", "error", "critical"):
                    prefix = "⚠" if sev == "warning" else "❌"
                    lines.append(f"  {prefix} {title}")
            lines.append("--------------------------------------------")

        # Health & Summary Stats
        status_label = "🟢 READY FOR TRAINING" if self.ready_for_training else "🔴 NOT READY FOR TRAINING"
        lines.extend(
            [
                "Summary",
                f"Health          {self.health_score} / 100 ({self.health_grade})",
                f"Passed Checks   {self.passed_checks} / {self.total_checks}",
                f"Warnings        {self.warning_count}",
                f"Errors          {self.error_count}",
                f"Execution Time  {self.execution_time:.2f} sec",
                f"Status          {status_label}",
                "══════════════════════════════════════════════",
            ]
        )

        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f" • {rec}")

        return "\n".join(lines)

    def print(self) -> None:
        """Print the validation summary text to stdout."""
        print(self.summary_text())

    def to_dict(self) -> dict[str, Any]:
        """Convert validation summary into a dictionary."""
        return {
            "passed": self.passed,
            "health_score": self.health_score,
            "health_grade": self.health_grade,
            "health_rating": self.health_rating,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "warning_count": self.warning_count,
            "error_count": self.error_count,
            "critical_count": self.critical_count,
            "execution_time": self.execution_time,
            "validator_results": self.validator_results,
            "recommendations": self.recommendations,
            "ready_for_training": self.ready_for_training,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize validation summary to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
