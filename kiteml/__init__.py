"""
KiteML - A lightweight AutoML library for quick model training and evaluation.
"""

__version__ = "1.0.2"
__author__ = "KiteML Team"

from kiteml.config import DEFAULT_CV_FOLDS, DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE
from kiteml.ml import TrainingResult, load, train
from kiteml.orchestration import KiteMLPipeline, PipelineBuildResult
from kiteml.output.result import (
    ClassificationMetrics,
    RegressionMetrics,
    Result,
    TrainingTimes,
)
from kiteml.validation import validate

__all__ = [
    "train",
    "load",
    "validate",
    "TrainingResult",
    "Result",
    "ClassificationMetrics",
    "RegressionMetrics",
    "TrainingTimes",
    "KiteMLPipeline",
    "PipelineBuildResult",
    "DEFAULT_TEST_SIZE",
    "DEFAULT_RANDOM_STATE",
    "DEFAULT_CV_FOLDS",
]
