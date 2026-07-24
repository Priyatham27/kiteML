"""
session.py — TrainingSession and TrainingResult data models for KiteML training engine.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from kiteml.training.context import TrainingContext
from kiteml.training.metrics import TrainingMetrics


@dataclass
class TrainingSession:
    """
    Metadata container for a training session run.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    status: str = "PENDING"
    task_type: str = "unknown"


@dataclass
class TrainingResult:
    """
    Result container returned by TrainingEngine.train().
    """

    session: TrainingSession
    context: TrainingContext
    metrics: TrainingMetrics
    experiment_metadata: dict[str, Any] = field(default_factory=dict)
    fitted_model: Any = None

    def summary(self, width: int = 55) -> str:
        """Render terminal summary box of training result."""
        lines = [
            "═" * width,
            "🎯 KiteML Training Session Result",
            "═" * width,
            f"  Session ID    {self.session.session_id}",
            f"  Task Type     {self.session.task_type}",
            f"  Status        {self.session.status}",
            f"  Training Time {self.metrics.training_time:.3f} sec",
            f"  Samples       {self.metrics.n_samples} rows × {self.metrics.n_features} cols",
            f"  CV Folds      {self.metrics.n_folds}",
            "═" * width,
        ]
        return "\n".join(lines)
