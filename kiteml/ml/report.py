"""
report.py — UnifiedReport summary renderer for KiteML Unified ML Engine.
"""

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UnifiedReport:
    """
    Unified summary report combining dataset intelligence, evaluation, and selection rankings.
    """

    model_name: str
    problem_type: str
    composite_score: float
    explanation: str
    stage_timings: dict[str, float] = field(default_factory=dict)
    total_time_sec: float = 0.0

    def summary(self, width: int = 55) -> str:
        """Render terminal summary dashboard."""
        lines = [
            "═" * width,
            "🪁 KiteML Unified Training Result Summary",
            "═" * width,
            f"  Winner Model     {self.model_name}",
            f"  Problem Type     {self.problem_type.capitalize()}",
            f"  Composite Score  {self.composite_score:.2f} / 100",
            f"  Execution Time   {self.total_time_sec:.2f} s",
            "─" * width,
            f"  Explanation: {self.explanation}",
            "═" * width,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to dictionary."""
        return {
            "winner": self.model_name,
            "problem_type": self.problem_type,
            "composite_score": self.composite_score,
            "explanation": self.explanation,
            "stage_timings": self.stage_timings,
            "total_time_sec": self.total_time_sec,
        }

    def to_json(self) -> str:
        """Export report to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
