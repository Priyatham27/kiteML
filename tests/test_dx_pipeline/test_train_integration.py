"""
test_train_integration.py — Integration tests for DXPipeline in train() and Result.diagnostics() (Story 3.6).
"""

import pandas as pd
import pytest

import kiteml
from kiteml.pipeline import create_dx_pipeline


def test_dx_pipeline_creation_and_run():
    dx = create_dx_pipeline()
    dx.set_validation_status("Passed")
    dx.set_training_status("Completed")

    diag = dx.get_diagnostics()
    assert diag.validation_status == "Passed"
    assert diag.training_status == "Completed"


def test_result_diagnostics_api():
    df = pd.DataFrame(
        {
            "feature1": list(range(20)),
            "feature2": [1] * 20,
            "target": [0, 1] * 10,
        }
    )

    res = kiteml.train(df, target="target", validate_data=True)

    assert hasattr(res, "diagnostics")
    diag_summary = res.diagnostics()
    assert "KiteML Diagnostics" in diag_summary
    assert "Execution Time" in diag_summary
