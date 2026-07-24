"""
test_report.py — Unit tests for UnifiedReport (Story 5.8).
"""

import pytest

from kiteml.ml import UnifiedReport


def test_unified_report():
    report = UnifiedReport(
        model_name="RandomForest",
        problem_type="classification",
        composite_score=94.5,
        explanation="Winning model",
    )

    assert "RandomForest" in report.summary()
    assert report.to_dict()["composite_score"] == 94.5
