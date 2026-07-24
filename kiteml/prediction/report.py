"""
report.py — PredictionReport data model and terminal summary formatting for KiteML.
"""

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PredictionReport:
    """
    Report metadata for a prediction run.
    """

    n_samples: int
    has_probabilities: bool = False
    average_confidence: float | None = None
    execution_time_sec: float = 0.0

    def summary(self, width: int = 55) -> str:
        """Render visual terminal summary box."""
        lines = [
            "═" * width,
            "🔮 KiteML Prediction Execution Summary",
            "═" * width,
            f"  Samples Predicted  {self.n_samples}",
            f"  Probabilities      {'Yes' if self.has_probabilities else 'No'}",
        ]
        if self.average_confidence is not None:
            lines.append(f"  Avg Confidence     {self.average_confidence * 100.0:.2f}%")
        lines.append(f"  Execution Time     {self.execution_time_sec * 1000.0:.2f} ms")
        lines.append("═" * width)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to dictionary."""
        return {
            "n_samples": self.n_samples,
            "has_probabilities": self.has_probabilities,
            "average_confidence": self.average_confidence,
            "execution_time_sec": self.execution_time_sec,
        }

    def to_json(self) -> str:
        """Export report to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
