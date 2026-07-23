"""
test_warning.py — Unit tests for KiteMLWarning base and domain subclasses (Story 3.4).
"""

import pytest

from kiteml.warnings import (
    DatasetWarning,
    DeploymentWarning,
    KiteMLWarning,
    PerformanceWarning,
    PredictionWarning,
    SchemaWarning,
    TrainingWarning,
    ValidationWarning,
    WarningSeverity,
)


def test_warning_initialization():
    w = KiteMLWarning(
        message="High missing values detected",
        code="KML-W-D001",
        severity=WarningSeverity.HIGH,
        recommendation="Impute values using median",
        category="Dataset",
    )

    assert isinstance(w, UserWarning)
    assert w.message == "High missing values detected"
    assert w.code == "KML-W-D001"
    assert w.severity == WarningSeverity.HIGH
    assert w.recommendation == "Impute values using median"
    assert "KML-W-D001" in str(w)
    assert "Recommendation" in str(w)


def test_domain_warning_subclasses():
    warnings_list = [
        (DatasetWarning("ds warn"), "Dataset"),
        (SchemaWarning("schema warn"), "Schema"),
        (ValidationWarning("val warn"), "Validation"),
        (TrainingWarning("train warn"), "Training"),
        (PredictionWarning("pred warn"), "Prediction"),
        (DeploymentWarning("deploy warn"), "Deployment"),
        (PerformanceWarning("perf warn"), "Performance"),
    ]

    for w_obj, expected_cat in warnings_list:
        assert isinstance(w_obj, KiteMLWarning)
        assert w_obj.category == expected_cat
