"""
best_model.py — BestModel container data model for selected optimal algorithm in KiteML.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BestModel:
    """
    Container representing the single optimal model selected by KiteML ModelSelectionEngine.
    """

    model: Any
    model_name: str
    pipeline: Any | None = None
    evaluation: Any | None = None
    composite_score: float = 0.0
    explanation: str = ""
    pareto_frontier: dict[str, Any] = field(default_factory=dict)

    def predict(self, X: Any) -> Any:
        """Convenience method delegating prediction to underlying model."""
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        """Convenience method delegating probability prediction if supported."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise AttributeError(f"Model '{self.model_name}' does not support predict_proba().")

    def summary(self, width: int = 55) -> str:
        """Render terminal summary box of selected best model."""
        lines = [
            "═" * width,
            "🏆 KiteML Optimal Selected Model",
            "═" * width,
            f"  Winner Model     {self.model_name}",
            f"  Composite Score  {self.composite_score:.2f} / 100",
            "─" * width,
            f"  Explanation: {self.explanation}",
            "═" * width,
        ]
        return "\n".join(lines)
