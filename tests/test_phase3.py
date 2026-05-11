"""
Tests for Phase 3: Production & Deployment Layer.

Tests:
- Model Packaging & Serialization
- Inference Guardrails & Realtime Inference
- Batch Prediction
- Drift Detection
- Experiment Tracking & Lineage
- Docker & ONNX Export Generation
"""

import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from kiteml.core import train
from kiteml.deployment.packaging import load_bundle
from kiteml.experiments.tracker import list_runs


@pytest.fixture(scope="module")
def iris_data():
    iris = load_iris(as_frame=True)
    df = iris.frame
    # Rename for cleaner assertions
    df.columns = ["sl", "sw", "pl", "pw", "target"]
    return df


@pytest.fixture(scope="module")
def trained_result(iris_data):
    return train(iris_data, target="target", problem_type="classification", verbose=False)


class TestPackagingAndDeployment:

    def test_bundle_packaging_and_loading(self, trained_result, tmp_path):
        """Test creating and loading a complete .kiteml bundle."""
        bundle_path = str(tmp_path / "model.kiteml")
        
        # Package
        pkg_res = trained_result.package(path=bundle_path, target_column="target")
        assert os.path.exists(bundle_path)
        assert pkg_res.total_size_bytes > 0
        
        # Verify files
        files = os.listdir(bundle_path)
        assert "model.joblib" in files
        assert "metadata.json" in files
        assert "manifest.yaml" in files
        assert "schema.json" in files
        
        # Load
        bundle = load_bundle(bundle_path)
        assert "model" in bundle
        assert "metadata" in bundle
        assert bundle["metadata"]["problem_type"] == "classification"
        assert bundle["metadata"]["n_features"] == 4


    def test_inference_guardrails(self, trained_result):
        """Test guardrails catch schema mismatches."""
        from kiteml.deployment.inference_guardrails import InferenceGuardrails
        
        guard = InferenceGuardrails(trained_result.feature_names)
        
        # Missing column
        bad_df = pd.DataFrame({"sl": [5.1], "sw": [3.5], "pl": [1.4]}) # missing pw
        res = guard.validate(bad_df)
        assert not res.is_valid
        assert any(v.violation_type == "missing_column" for v in res.errors)
        
        with pytest.raises(ValueError):
            res.raise_if_invalid()
            
        # Extra column (warning, not error if allowed)
        guard_strict = InferenceGuardrails(trained_result.feature_names, allow_extra_columns=False)
        extra_df = pd.DataFrame({"sl": [5.1], "sw": [3.5], "pl": [1.4], "pw": [0.2], "extra": [99]})
        res = guard_strict.validate(extra_df)
        assert res.is_valid  # Warnings don't fail validation
        assert any(v.violation_type == "extra_column" for v in res.warnings)


    def test_realtime_inference(self, trained_result, iris_data):
        """Test fast single-record inference."""
        engine = trained_result.realtime_engine()
        
        # Dict input
        row = {"sl": 5.1, "sw": 3.5, "pl": 1.4, "pw": 0.2}
        pred = engine.predict(row)
        
        assert pred.prediction is not None
        assert pred.latency_ms > 0
        assert isinstance(pred.probabilities, dict) # Classification probas


    def test_batch_inference(self, trained_result, iris_data, tmp_path):
        """Test chunked batch prediction."""
        # Drop target
        X = iris_data.drop(columns=["target"])
        
        res = trained_result.batch_predict(X, chunk_size=50, verbose=False)
        assert res.n_rows == 150
        assert res.n_chunks == 3
        assert len(res.predictions) == 150
        assert res.probabilities is not None
        assert res.probabilities.shape == (150, 3)
        
        # Test CSV export
        csv_path = str(tmp_path / "preds.csv")
        res.save_csv(csv_path)
        assert os.path.exists(csv_path)


class TestMonitoring:

    def test_drift_monitor(self, trained_result, iris_data):
        """Test data drift detection."""
        X_ref = iris_data.drop(columns=["target"])
        
        # Create heavily drifted data (multiply sepal length by 10)
        X_drift = X_ref.copy()
        X_drift["sl"] = X_drift["sl"] * 10
        
        report = trained_result.monitor_drift(current_data=X_drift, reference_data=X_ref)
        
        assert report.drift_detected
        assert "sl" in report.drifted_features
        assert report.feature_results["sl"].psi > 0.2
        assert report.overall_psi > 0
        assert report.severity in ("moderate", "high")


    def test_anomaly_monitor(self, trained_result, iris_data):
        """Test detection of anomalous individual rows."""
        from kiteml.monitoring.anomaly_monitor import AnomalyMonitor
        
        X_ref = iris_data.drop(columns=["target"])
        monitor = AnomalyMonitor.from_result(trained_result, X_ref, method="zscore", zscore_threshold=3.0)
        
        # Normal data
        res1 = monitor.check(X_ref.head(10))
        assert not res1.has_anomalies
        
        # Anomalous data
        X_anom = X_ref.head(5).copy()
        X_anom.loc[2, "sl"] = 999.0  # Massive outlier
        
        res2 = monitor.check(X_anom)
        assert res2.has_anomalies
        assert res2.n_anomalous == 1
        assert 2 in res2.anomalous_rows
        assert "sl" in res2.feature_violation_counts


class TestGovernanceAndExperiments:

    def test_experiment_tracking(self, trained_result, iris_data, tmp_path):
        """Test run tracking and metadata persistence."""
        from kiteml.experiments.tracker import _DEFAULT_STORE
        import kiteml.experiments.tracker as tracker
        
        # Override store path for test
        tracker._DEFAULT_STORE = str(tmp_path / "exp_store")
        
        run = trained_result.experiment(
            experiment_name="iris_test",
            dataset=iris_data,
            tags={"env": "test"}
        )
        
        assert run.run_id is not None
        assert run.model_name == trained_result.model_name
        assert run.tags["env"] == "test"
        
        runs = list_runs("iris_test", store_path=tracker._DEFAULT_STORE)
        assert len(runs) == 1
        assert runs[0].run_id == run.run_id


    def test_lineage_tracking(self, trained_result, iris_data):
        """Test full pipeline lineage generation."""
        lin = trained_result.lineage(dataset=iris_data, print_tree=False)
        
        assert lin.lineage_id is not None
        assert len(lin.steps) == 5  # data, prep, selection, train, eval
        assert lin.steps[0].step_type == "data"
        assert "dataset_hash" in lin.steps[0].artifacts
        assert lin.steps[4].step_type == "evaluation"


    def test_versioning(self, trained_result, tmp_path):
        """Test semantic model versioning."""
        import kiteml.governance.versioning as ver
        ver._DEFAULT_REGISTRY = str(tmp_path / "versions")
        
        v1 = trained_result.version()
        assert v1.version == "v1.0.0"
        
        v2 = trained_result.version(bump="minor")
        assert v2.version == "v1.1.0"
        
        v3 = trained_result.version(bump="major")
        assert v3.version == "v2.0.0"


class TestExportGeneration:

    def test_docker_export(self, trained_result, tmp_path):
        """Test generation of Docker deployment artifacts."""
        out_dir = str(tmp_path / "docker")
        res = trained_result.export_docker(output_dir=out_dir)
        
        assert os.path.isdir(out_dir)
        files = os.listdir(out_dir)
        assert "Dockerfile" in files
        assert "requirements.txt" in files
        assert "serve.py" in files
        assert "docker-compose.yml" in files


    def test_api_generator(self, trained_result, tmp_path):
        """Test generation of standalone FastAPI script."""
        out_dir = str(tmp_path / "api")
        path = trained_result.generate_api(output_dir=out_dir)
        
        assert os.path.exists(path)
        with open(path, "r") as f:
            content = f.read()
            assert "FastAPI" in content
            assert trained_result.model_name in content


    def test_dashboard_generation(self, trained_result, tmp_path):
        """Test generation of HTML deployment dashboard."""
        dash_path = str(tmp_path / "dash.html")
        trained_result.generate_dashboard(path=dash_path)
        
        assert os.path.exists(dash_path)
        with open(dash_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "KiteML Production Dashboard" in content
            assert "Deployment Readiness" in content
