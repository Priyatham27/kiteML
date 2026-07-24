"""
report.py — SelectionReport leaderboard summary renderer for KiteML model selection.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from kiteml.selection.best_model import BestModel


@dataclass
class SelectionReport:
    """
    Model selection report with leaderboard rankings and Pareto frontier.
    """

    best_model: BestModel
    leaderboard: list[dict[str, Any]] = field(default_factory=list)
    pareto_frontier: dict[str, Any] = field(default_factory=dict)
    policy: str = "balanced"

    def summary(self, width: int = 55) -> str:
        """Render terminal leaderboard summary."""
        lines = [
            "═" * width,
            f"👑 KiteML Model Leaderboard (Policy: {self.policy})",
            "═" * width,
        ]
        for item in self.leaderboard:
            rank = item.get("rank", 1)
            name = item.get("name", "Unknown")
            score = item.get("policy_score", item.get("composite_score", 0.0))
            badge = " 🏆 (Winner)" if rank == 1 else ""
            lines.append(f"  #{rank:<2} {name:<24} Score: {score:.2f}{badge}")

        lines.append("═" * width)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize selection report to dictionary."""
        return {
            "winner": self.best_model.model_name,
            "policy": self.policy,
            "leaderboard": self.leaderboard,
            "pareto_frontier": self.pareto_frontier,
            "explanation": self.best_model.explanation,
        }

    def to_json(self) -> str:
        """Export selection report to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
