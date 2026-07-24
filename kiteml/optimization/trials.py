"""
trials.py — OptimizationTrial and TrialManager data models for tracking tuning experiments in KiteML.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OptimizationTrial:
    """
    Data model representing a single hyperparameter evaluation trial.
    """

    trial_id: int
    parameters: dict[str, Any]
    score: float = 0.0
    duration: float = 0.0
    status: str = "COMPLETED"


class TrialManager:
    """
    Manages and records optimization trials.
    """

    def __init__(self) -> None:
        self.trials: list[OptimizationTrial] = []

    def record_trial(self, trial: OptimizationTrial) -> None:
        """Record completed trial."""
        self.trials.append(trial)

    def get_best_trial(self) -> OptimizationTrial | None:
        """Retrieve trial with highest score."""
        if not self.trials:
            return None
        return max(self.trials, key=lambda t: t.score)
