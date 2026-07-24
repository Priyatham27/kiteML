"""
context.py — OptimizationContext shared state data model for KiteML hyperparameter optimization.
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from kiteml.optimization.search_space import SearchSpace


@dataclass
class OptimizationContext:
    """
    Shared context container passing parameters, search space, and dataset across optimization steps.
    """

    dataframe: pd.DataFrame | None = None
    target_name: str | None = None
    model_name: str = "unknown"
    search_space: SearchSpace | None = None
    best_parameters: dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
