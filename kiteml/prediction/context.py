"""
context.py — PredictionContext shared state model for KiteML prediction engine.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class PredictionContext:
    """
    Shared state container passing input dataset, pipeline, and raw outputs.
    """

    dataframe: pd.DataFrame | None = None
    adapted_dataframe: pd.DataFrame | None = None
    transformed_matrix: Any = None
    model: Any | None = None
    pipeline: Any | None = None
    predictions: Any = None
    probabilities: Any = None
