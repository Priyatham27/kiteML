"""
KiteML - A lightweight AutoML library for quick model training and evaluation.
"""

__version__ = "0.1.0"
__author__ = "KiteML Team"

from kiteml.core import train
from kiteml.config import DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE, DEFAULT_CV_FOLDS
from kiteml.output.result import (
    Result,
    ClassificationMetrics,
    RegressionMetrics,
    TrainingTimes,
)
