"""
test_manifest.py — Unit tests for DeploymentManifest (Story 5.7).
"""

import pytest

from kiteml.deployment import DeploymentManifest


def test_deployment_manifest():
    manifest = DeploymentManifest(
        model_name="RandomForestClassifier",
        task_type="classification",
        checksum="abcd1234sha256",
    )

    d = manifest.to_dict()
    assert d["model_name"] == "RandomForestClassifier"
    assert d["task_type"] == "classification"
    assert d["checksum"] == "abcd1234sha256"
    assert d["kiteml_version"] == "1.0.2"
