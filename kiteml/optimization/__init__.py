"""
optimization/ — Intelligent Hyperparameter Optimization Engine package for KiteML.
"""

from kiteml.optimization.advisor import OptimizationAdvisor
from kiteml.optimization.context import OptimizationContext
from kiteml.optimization.early_stopping import EarlyStopping
from kiteml.optimization.engine import OptimizationEngine
from kiteml.optimization.metrics import OptimizationMetrics
from kiteml.optimization.search_space import SearchSpace
from kiteml.optimization.session import OptimizationResult, OptimizationSession
from kiteml.optimization.strategies import (
    GridSearchStrategy,
    OptimizationStrategy,
    RandomSearchStrategy,
)
from kiteml.optimization.trials import OptimizationTrial, TrialManager

__all__ = [
    "OptimizationEngine",
    "OptimizationAdvisor",
    "SearchSpace",
    "OptimizationStrategy",
    "GridSearchStrategy",
    "RandomSearchStrategy",
    "EarlyStopping",
    "OptimizationTrial",
    "TrialManager",
    "OptimizationSession",
    "OptimizationResult",
    "OptimizationMetrics",
    "OptimizationContext",
]
