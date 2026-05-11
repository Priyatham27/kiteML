"""
models — Model registry, selection, and model collections for KiteML.
"""

from kiteml.models.registry import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    get_classification_models,
    get_regression_models,
)
from kiteml.models.selector import select_best_model

__all__ = [
    "CLASSIFICATION_MODELS",
    "REGRESSION_MODELS",
    "get_classification_models",
    "get_regression_models",
    "select_best_model",
]
