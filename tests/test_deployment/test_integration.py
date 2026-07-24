"""
test_integration.py — Integration tests for DeploymentEngine (Story 5.7).
"""

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from kiteml.deployment import DeploymentEngine


def test_deployment_engine_save_load_predict_roundtrip(tmp_path):
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    y = pd.Series([0, 1, 0, 1])

    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)

    engine = DeploymentEngine()
    pkg_path = tmp_path / "rf.kiteml"

    engine.package(
        model=rf,
        model_name="RandomForestClassifier",
        task_type="classification",
        output_path=pkg_path,
        feature_names=list(X.columns),
    )

    loaded = engine.load(pkg_path)
    preds = loaded.predict(X)

    assert len(preds) == 4
