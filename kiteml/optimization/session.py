"""
session.py — OptimizationSession and OptimizationResult models for KiteML hyperparameter tuning.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from kiteml.optimization.metrics import OptimizationMetrics
from kiteml.optimization.trials import OptimizationTrial


@dataclass
class OptimizationSession:
    """
    Session metadata container for hyperparameter optimization search.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_name: str = "unknown"
    strategy_name: str = "unknown"
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    status: str = "PENDING"


@dataclass
class OptimizationResult:
    """
    Result container returned by OptimizationEngine.optimize().
    """

    best_parameters: dict[str, Any]
    best_score: float
    best_trial: OptimizationTrial | None
    session: OptimizationSession
    trials: list[OptimizationTrial]
    metrics: OptimizationMetrics
