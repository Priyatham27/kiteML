"""
metrics.py — TrainingMetrics model for recording runtime statistics in KiteML.
"""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TrainingMetrics:
    """
    Runtime execution metrics recorded during a training session.
    """

    training_time: float = 0.0
    cpu_time: float = 0.0
    n_samples: int = 0
    n_features: int = 0
    n_folds: int = 5

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to dictionary."""
        return asdict(self)
