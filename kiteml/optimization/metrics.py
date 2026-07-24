"""
metrics.py — OptimizationMetrics model for recording hyperparameter tuning metrics in KiteML.
"""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class OptimizationMetrics:
    """
    Performance metrics recorded during optimization search.
    """

    n_trials: int = 0
    best_score: float = 0.0
    optimization_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to dictionary."""
        return asdict(self)
