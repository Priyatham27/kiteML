"""
context.py — EvaluationContext shared state model for KiteML evaluation engine.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class EvaluationContext:
    """
    Shared evaluation state holding model, test set, and predictions.
    """

    model: Any | None = None
    X_test: pd.DataFrame | Any = None
    y_test: pd.Series | Any = None
    y_pred: Any = None
    y_proba: Any = None
    task_type: str = "classification"
