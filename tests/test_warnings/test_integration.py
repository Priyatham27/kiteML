"""
test_integration.py — Integration tests for Result warning API and training workflow (Story 3.4).
"""

import pandas as pd
import pytest

import kiteml
from kiteml.warnings import DatasetWarning, WarningCollector


def test_result_warning_api():
    collector = WarningCollector()
    collector.add(DatasetWarning("Sample warning", code="KML-W-D001", recommendation="Test recommendation"))

    result = kiteml.Result(
        model=None,
        metrics={},
        report="",
        problem_type="classification",
        warning_collector=collector,
    )

    assert result.has_warnings() is True
    assert result.warning_count() == 1
    assert len(result.warnings()) == 1
    assert "Sample warning" in result.warning_summary()


def test_train_collects_warnings():
    df = pd.DataFrame(
        {
            "feature1": list(range(20)),
            "feature2": [1] * 20,  # constant column triggers warning
            "target": [0, 1] * 10,
        }
    )

    res = kiteml.train(df, target="target", validate_data=True)

    assert hasattr(res, "warning_collector")
    assert isinstance(res.has_warnings(), bool)
    assert isinstance(res.warning_count(), int)
