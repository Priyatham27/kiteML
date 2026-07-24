"""
report.py — EvaluationReport data model and terminal summary formatting for KiteML.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from kiteml.evaluation.benchmark import BenchmarkMetrics


@dataclass
class EvaluationReport:
    """
    Comprehensive evaluation report container.
    """

    task_type: str
    composite_score: float
    metrics: dict[str, Any] = field(default_factory=dict)
    benchmark: BenchmarkMetrics = field(default_factory=BenchmarkMetrics)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def summary(self, width: int = 55) -> str:
        """Render visual summary box of evaluation report."""
        lines = [
            "═" * width,
            "📈 KiteML Model Evaluation Report",
            "═" * width,
            f"  Task Type        {self.task_type}",
            f"  Composite Score  {self.composite_score:.2f} / 100",
            "─" * width,
            "  Primary Metrics:",
        ]

        for k, v in self.metrics.items():
            if k != "confusion_matrix":
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                lines.append(f"   ► {k:<18} : {val_str}")

        lines.append("─" * width)
        lines.append(f"  Inference Latency {self.benchmark.inference_latency_ms:.3f} ms / sample")
        lines.append(f"  Model Footprint   {self.benchmark.model_size_kb:.2f} KB")
        lines.append("═" * width)

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize evaluation report to dictionary."""
        return {
            "task_type": self.task_type,
            "composite_score": self.composite_score,
            "metrics": self.metrics,
            "benchmark": self.benchmark.to_dict(),
            "diagnostics": self.diagnostics,
        }

    def to_json(self) -> str:
        """Export report to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def generate_report(
    metrics: Any,
    problem_type: str = "classification",
    model_name: str = "Model",
    all_results: Any = None,
) -> str:
    """
    Legacy helper formatting evaluation metrics into a report string.
    """
    lines = [
        "==================================================",
        f"       KiteML Evaluation Report: {model_name}",
        "==================================================",
        f"Problem Type: {problem_type.capitalize()}",
    ]
    if hasattr(metrics, "to_dict"):
        m_dict = metrics.to_dict()
    elif isinstance(metrics, dict):
        m_dict = metrics
    else:
        m_dict = {}

    pretty_names = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1 Score",
        "f1": "F1 Score",
        "r2_score": "R2 Score",
        "r2": "R2 Score",
        "rmse": "RMSE",
        "mse": "MSE",
        "mae": "MAE",
    }

    for k, v in m_dict.items():
        if k in ("confusion_matrix", "classification_report"):
            continue
        label = pretty_names.get(k, k.replace("_", " ").title())
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        lines.append(f"{label:<20}: {val_str}")

    if all_results:
        lines.append("--------------------------------------------------")
        lines.append("Model Leaderboard:")
        for name, res in all_results.items():
            if isinstance(res, dict):
                score = res.get("score", 0.0)
                rank = res.get("rank", 1)
                lines.append(f" #{rank} {name:<20}: {score:.4f}")
            else:
                lines.append(f" - {name:<20}: {res}")

    lines.append("==================================================")
    return "\n".join(lines)
