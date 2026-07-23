"""
report.py — WarningReport class for aggregated warning summaries in KiteML.
"""

from dataclasses import dataclass, field
from typing import Any

from kiteml.warnings.base import KiteMLWarning
from kiteml.warnings.severity import WarningSeverity, get_warning_icon


@dataclass
class WarningReport:
    """
    Aggregated warning summary report.

    Attributes
    ----------
    warnings : list of KiteMLWarning
        List of collected warnings.
    """

    warnings: list[KiteMLWarning] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        """Total number of warnings."""
        return len(self.warnings)

    @property
    def by_category(self) -> dict[str, int]:
        """Count of warnings grouped by category."""
        counts: dict[str, int] = {}
        for w in self.warnings:
            counts[w.category] = counts.get(w.category, 0) + 1
        return counts

    @property
    def by_severity(self) -> dict[str, int]:
        """Count of warnings grouped by severity level."""
        counts: dict[str, int] = {}
        for w in self.warnings:
            sev = w.severity.value if isinstance(w.severity, WarningSeverity) else str(w.severity)
            counts[sev] = counts.get(sev, 0) + 1
        return counts

    def summary_text(self, width: int = 50) -> str:
        """Generate formatted summary report text."""
        if not self.warnings:
            return "✅ No warnings detected."

        lines = [
            "═" * width,
            "⚠ KiteML Warning Summary",
            "═" * width,
        ]

        # Category breakdown
        for cat_name, count in self.by_category.items():
            lines.append(f"  {cat_name:<25} {count} warning(s)")

        lines.append("─" * width)

        # Warning items
        for w in self.warnings:
            icon = get_warning_icon(w.severity)
            lines.append(f"  {icon} {w.message}")
            if w.recommendation:
                lines.append(f"     → Recommendation: {w.recommendation}")
            lines.append("")

        lines.append("═" * width)
        lines.append("Training Continued Successfully")
        lines.append("═" * width)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize WarningReport to dictionary."""
        return {
            "total_count": self.total_count,
            "by_category": self.by_category,
            "by_severity": self.by_severity,
            "warnings": [w.to_dict() for w in self.warnings],
        }
