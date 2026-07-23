"""
test_warning_pipeline.py — Unit tests for WarningManager (Story 3.6).
"""

import pytest

from kiteml.pipeline.warning_manager import WarningManager
from kiteml.warnings import DatasetWarning


def test_warning_manager():
    mgr = WarningManager()
    mgr.add(DatasetWarning("High missing values", code="KML-W-D001"))

    assert len(mgr.warnings) == 1
    report = mgr.get_report()
    assert report.total_count == 1
    assert "High missing values" in mgr.format_report()
