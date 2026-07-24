"""
training/ — Intelligent Model Training Engine package for KiteML.
"""

from kiteml.training.context import TrainingContext
from kiteml.training.cross_validation import CrossValidationEngine
from kiteml.training.dag import TrainingDAG, TrainingNode
from kiteml.training.engine import TrainingEngine
from kiteml.training.lifecycle import TrainingLifecycle
from kiteml.training.metrics import TrainingMetrics
from kiteml.training.session import TrainingResult, TrainingSession
from kiteml.training.splitter import DataSplitter
from kiteml.training.state import TrainingState
from kiteml.training.task_detector import TaskDetector
from kiteml.training.tracker import ExperimentTracker
from kiteml.training.trainer import ModelTrainer, train_model

__all__ = [
    "TrainingEngine",
    "TaskDetector",
    "DataSplitter",
    "CrossValidationEngine",
    "TrainingState",
    "TrainingLifecycle",
    "TrainingContext",
    "TrainingMetrics",
    "ExperimentTracker",
    "TrainingDAG",
    "TrainingNode",
    "ModelTrainer",
    "train_model",
    "TrainingSession",
    "TrainingResult",
]
