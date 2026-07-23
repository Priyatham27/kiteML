"""
warning.py — Domain-specific warning subclasses for KiteML.
"""

from typing import Any

import kiteml.warnings.categories as cat
from kiteml.warnings.base import KiteMLWarning
from kiteml.warnings.severity import WarningSeverity


class DatasetWarning(KiteMLWarning):
    """Warning raised for dataset structure or content non-fatal issues."""

    def __init__(
        self,
        message: str,
        code: str = "KML-W-D000",
        severity: WarningSeverity | str = WarningSeverity.MEDIUM,
        recommendation: str | None = None,
        context: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            severity=severity,
            recommendation=recommendation,
            context=context,
            category=cat.DATASET,
            source=source,
        )


class SchemaWarning(KiteMLWarning):
    """Warning raised for feature schema or column structure issues."""

    def __init__(
        self,
        message: str,
        code: str = "KML-W-S000",
        severity: WarningSeverity | str = WarningSeverity.MEDIUM,
        recommendation: str | None = None,
        context: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            severity=severity,
            recommendation=recommendation,
            context=context,
            category=cat.SCHEMA,
            source=source,
        )


class ValidationWarning(KiteMLWarning):
    """Warning raised for dataset quality or validation rule non-fatal issues."""

    def __init__(
        self,
        message: str,
        code: str = "KML-W-V000",
        severity: WarningSeverity | str = WarningSeverity.MEDIUM,
        recommendation: str | None = None,
        context: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            severity=severity,
            recommendation=recommendation,
            context=context,
            category=cat.VALIDATION,
            source=source,
        )


class TrainingWarning(KiteMLWarning):
    """Warning raised for model training or cross-validation non-fatal issues."""

    def __init__(
        self,
        message: str,
        code: str = "KML-W-M000",
        severity: WarningSeverity | str = WarningSeverity.MEDIUM,
        recommendation: str | None = None,
        context: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            severity=severity,
            recommendation=recommendation,
            context=context,
            category=cat.TRAINING,
            source=source,
        )


class PredictionWarning(KiteMLWarning):
    """Warning raised during inference for feature drift, unknown categories, etc."""

    def __init__(
        self,
        message: str,
        code: str = "KML-W-I000",
        severity: WarningSeverity | str = WarningSeverity.MEDIUM,
        recommendation: str | None = None,
        context: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            severity=severity,
            recommendation=recommendation,
            context=context,
            category=cat.PREDICTION,
            source=source,
        )


class DeploymentWarning(KiteMLWarning):
    """Warning raised for export, bundle, or dependency non-fatal issues."""

    def __init__(
        self,
        message: str,
        code: str = "KML-W-DP000",
        severity: WarningSeverity | str = WarningSeverity.MEDIUM,
        recommendation: str | None = None,
        context: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            severity=severity,
            recommendation=recommendation,
            context=context,
            category=cat.DEPLOYMENT,
            source=source,
        )


class PerformanceWarning(KiteMLWarning):
    """Warning raised for slow training, high memory, or latency issues."""

    def __init__(
        self,
        message: str,
        code: str = "KML-W-P000",
        severity: WarningSeverity | str = WarningSeverity.MEDIUM,
        recommendation: str | None = None,
        context: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            severity=severity,
            recommendation=recommendation,
            context=context,
            category=cat.PERFORMANCE,
            source=source,
        )
