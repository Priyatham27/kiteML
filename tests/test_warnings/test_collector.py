"""
test_collector.py — Unit tests for WarningCollector (Story 3.4).
"""

import pytest

from kiteml.warnings import DatasetWarning, WarningCollector, WarningSeverity


def test_warning_collector_add_and_deduplication():
    collector = WarningCollector()
    w1 = DatasetWarning("High missing values", code="KML-W-D001")
    w2 = DatasetWarning("High missing values", code="KML-W-D001")

    assert collector.add(w1) is True
    assert collector.add(w2) is False  # Duplicate ignored
    assert len(collector) == 1


def test_collector_filtering():
    collector = WarningCollector()
    collector.warn("High missing", code="KML-W-D001", category="Dataset", severity=WarningSeverity.HIGH)
    collector.warn("Constant col", code="KML-W-S001", category="Schema", severity=WarningSeverity.MEDIUM)

    assert len(collector.get_by_category("Dataset")) == 1
    assert len(collector.get_by_severity(WarningSeverity.HIGH)) == 1
    assert len(collector.get_by_severity("HIGH")) == 1
