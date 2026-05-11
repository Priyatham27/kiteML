"""
KiteML - A lightweight AutoML library for quick model training and evaluation.
"""

__version__ = "0.1.0"
__author__ = "KiteML Team"

from kiteml.config import DEFAULT_CV_FOLDS, DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE
from kiteml.core import train
from kiteml.output.result import (
    ClassificationMetrics,
    RegressionMetrics,
    Result,
    TrainingTimes,
)
