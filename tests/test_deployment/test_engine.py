"""
test_engine.py — Unit tests for DeploymentEngine (Story 5.7).
"""

import pytest
from sklearn.dummy import DummyClassifier

from kiteml.deployment import DeploymentEngine, DeploymentReport, LoadedPackage


def test_deployment_engine_full_workflow(tmp_path):
    clf = DummyClassifier()
    clf.fit([[1]], [0])

    engine = DeploymentEngine()
    pkg_file = tmp_path / "model.kiteml"

    report = engine.package(
        model=clf,
        model_name="Dummy",
        task_type="classification",
        output_path=pkg_file,
    )

    assert isinstance(report, DeploymentReport)
    assert report.is_valid is True
    assert "📦 KiteML Package & Deployment Report" in report.summary()

    loaded = engine.load(pkg_file)
    assert isinstance(loaded, LoadedPackage)

    export_dir = engine.export(pkg_file, output_dir=tmp_path / "exported_fastapi", adapter="fastapi")
    assert (export_dir / "app.py").exists()
