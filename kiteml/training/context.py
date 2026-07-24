"""
context.py — TrainingContext shared state data model for KiteML training engine.
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class TrainingContext:
    """
    Shared context passing dataset, task type, split data, and metrics across training stages.
    """

    dataset: pd.DataFrame | None = None
    target_name: str | None = None
    task_type: str | None = None
    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_train: pd.Series | None = None
    y_test: pd.Series | None = None
    pipeline: Any | None = None
    models: list[Any] = field(default_factory=list)
    cv_scores: dict[str, list[float]] = field(default_factory=dict)
    metrics: Any | None = None
    diagnostics: Any | None = None
