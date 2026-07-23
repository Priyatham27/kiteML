"""
KiteML Warning Framework Package.

Intelligent warnings that educate without interrupting workflows.
"""

import kiteml.warnings.categories as categories
from kiteml.warnings.base import KiteMLWarning
from kiteml.warnings.collector import WarningCollector
from kiteml.warnings.formatter import WarningFormatter
from kiteml.warnings.policy import WarningPolicy
from kiteml.warnings.registry import (
    WarningDefinition,
    WarningRegistry,
    global_warning_registry,
)
from kiteml.warnings.report import WarningReport
from kiteml.warnings.severity import WarningSeverity, get_warning_icon
from kiteml.warnings.utils import emit_warning
from kiteml.warnings.warning import (
    DatasetWarning,
    DeploymentWarning,
    PerformanceWarning,
    PredictionWarning,
    SchemaWarning,
    TrainingWarning,
    ValidationWarning,
)

__all__ = [
    "KiteMLWarning",
    "DatasetWarning",
    "SchemaWarning",
    "ValidationWarning",
    "TrainingWarning",
    "PredictionWarning",
    "DeploymentWarning",
    "PerformanceWarning",
    "WarningSeverity",
    "get_warning_icon",
    "WarningDefinition",
    "WarningRegistry",
    "global_warning_registry",
    "WarningPolicy",
    "WarningCollector",
    "WarningReport",
    "WarningFormatter",
    "emit_warning",
    "categories",
]
