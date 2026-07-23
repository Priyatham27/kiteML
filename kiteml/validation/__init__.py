"""
KiteML Validation Framework Package.

Train only on validated data.
"""

from typing import Any

from kiteml.validation.dataset_validator import DatasetValidator
from kiteml.validation.engine import ValidationEngine
from kiteml.validation.feature_profile import FeatureProfile
from kiteml.validation.hooks import ValidationHookSystem
from kiteml.validation.message import ValidationMessage
from kiteml.validation.pipeline import ValidationPipeline
from kiteml.validation.quality_profile import QualityProfile
from kiteml.validation.quality_validator import QualityValidator
from kiteml.validation.registry import RuleRegistry, global_rule_registry
from kiteml.validation.rule import ValidationRule
from kiteml.validation.schema_validator import SchemaValidator
from kiteml.validation.severity import ValidationSeverity
from kiteml.validation.target_validator import TargetValidator
from kiteml.validation.utils import get_dataframe_memory_mb, timer
from kiteml.validation.validation_manager import ValidationManager
from kiteml.validation.validation_report import ValidationReport
from kiteml.validation.validation_result import ValidationResult
from kiteml.validation.validation_summary import ValidationSummary
from kiteml.validation.validator import BaseValidator


def validate(
    dataframe: Any,
    target: str | None = None,
    problem_type: str | None = None,
    fail_fast: bool = True,
    **kwargs: Any,
) -> ValidationSummary:
    """
    Validate a dataset using the full KiteML Validation Pipeline.

    Parameters
    ----------
    dataframe : Any
        Dataset DataFrame or file path to validate.
    target : str, optional
        Target column name.
    problem_type : str, optional
        'classification' or 'regression'.
    fail_fast : bool
        If True, stop pipeline immediately on critical failure.

    Returns
    -------
    ValidationSummary
    """
    pipeline = ValidationPipeline()
    return pipeline.validate(
        dataframe,
        target=target,
        problem_type=problem_type,
        fail_fast=fail_fast,
        **kwargs,
    )


__all__ = [
    "ValidationSeverity",
    "ValidationMessage",
    "ValidationResult",
    "ValidationRule",
    "BaseValidator",
    "DatasetValidator",
    "TargetValidator",
    "SchemaValidator",
    "QualityValidator",
    "FeatureProfile",
    "QualityProfile",
    "RuleRegistry",
    "global_rule_registry",
    "ValidationEngine",
    "ValidationReport",
    "ValidationManager",
    "ValidationPipeline",
    "ValidationSummary",
    "ValidationHookSystem",
    "validate",
    "get_dataframe_memory_mb",
    "timer",
]
