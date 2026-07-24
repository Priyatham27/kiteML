"""
test_statistics.py — Unit tests for TransformationStatistics (Story 4.6).
"""

import pytest

from kiteml.reporting import TransformationStatistics


def test_transformation_statistics_serialization():
    stats = TransformationStatistics(
        initial_rows=100,
        final_rows=100,
        initial_cols=10,
        final_cols=15,
        generated_features_count=5,
    )

    d = stats.to_dict()
    assert d["initial_rows"] == 100
    assert d["final_cols"] == 15
    assert d["generated_features_count"] == 5
