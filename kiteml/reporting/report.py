"""
report.py — StageReport and PipelineReport models for KiteML pipeline reporting.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kiteml.reporting.statistics import TransformationStatistics
from kiteml.reporting.timeline import TransformationTimeline


@dataclass
class StageReport:
    """Report for an individual pipeline execution stage."""

    stage_name: str
    duration: float = 0.0
    input_shape: tuple[int, int] = (0, 0)
    output_shape: tuple[int, int] = (0, 0)
    transformations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize stage report to dictionary."""
        return asdict(self)


@dataclass
class PipelineReport:
    """
    Comprehensive transformation pipeline report.
    """

    statistics: TransformationStatistics = field(default_factory=TransformationStatistics)
    timeline: TransformationTimeline = field(default_factory=TransformationTimeline)
    stage_reports: list[StageReport] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self, width: int = 55) -> str:
        """Render terminal visual summary of pipeline execution report."""
        stats = self.statistics
        lines = [
            "═" * width,
            "📊 KiteML Transformation Pipeline Report",
            "═" * width,
            f"  Initial Dataset Shape  {stats.initial_rows} rows × {stats.initial_cols} cols",
            f"  Final Dataset Shape    {stats.final_rows} rows × {stats.final_cols} cols",
            f"  Generated Features     {stats.generated_features_count}",
            f"  Dropped Features       {stats.dropped_features_count}",
            f"  Execution Time         {stats.total_execution_time:.3f} sec",
            "─" * width,
            "  Execution Timeline:",
        ]

        for evt in self.timeline.events:
            lines.append(
                f"   ► {evt.stage_name:<24} ({evt.duration_ms:5.1f} ms) {evt.input_shape} → {evt.output_shape}"
            )

        if self.warnings:
            lines.append("─" * width)
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings[:3]:
                lines.append(f"   ⚠ {w}")

        lines.append("═" * width)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete pipeline report to dictionary."""
        return {
            "statistics": self.statistics.to_dict(),
            "timeline": self.timeline.to_list(),
            "stage_reports": [s.to_dict() for s in self.stage_reports],
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "metadata": self.metadata,
        }

    def to_json(self, filepath: str | Path | None = None) -> str:
        """Export pipeline report to JSON."""
        json_str = json.dumps(self.to_dict(), indent=2)
        if filepath:
            out_path = Path(filepath)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json_str, encoding="utf-8")
        return json_str

    def to_html(self, filepath: str | Path | None = None) -> str:
        """Export pipeline report to HTML."""
        from kiteml.reporting.exporters import HTMLExporter

        return HTMLExporter().export(self, filepath=filepath)
