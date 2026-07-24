"""
test_descriptor.py — Unit tests for UniversalDeploymentDescriptor (Story 5.7 Flagship Feature).
"""

import pytest

from kiteml.deployment import UniversalDeploymentDescriptor


def test_universal_deployment_descriptor():
    descriptor = UniversalDeploymentDescriptor(
        model_name="XGBoostRegressor",
        task_type="regression",
        feature_names=["age", "income"],
    )

    d = descriptor.to_dict()
    assert d["model_name"] == "XGBoostRegressor"
    assert d["feature_names"] == ["age", "income"]
    assert "fastapi" in d["supported_adapters"]
