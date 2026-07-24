"""
state.py — TrainingState enum for tracking KiteML training progress.
"""

from enum import Enum


class TrainingState(str, Enum):
    """Lifecycle states for a training session."""

    CREATED = "CREATED"
    PREPARING = "PREPARING"
    SPLITTING = "SPLITTING"
    CROSS_VALIDATION = "CROSS_VALIDATION"
    TRAINING = "TRAINING"
    EVALUATION = "EVALUATION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
