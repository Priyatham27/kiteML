"""
test_report.py — Unit tests for WarningReport (Story 3.4).
"""

import pytest

from kiteml.warnings import DatasetWarning, SchemaWarning, WarningReport, WarningSeverity


def test_warning_report_aggregation():
    w1 = DatasetWarning("High missing values", code="KML-W-D001", severity=WarningSeverity.HIGH)
    w2 = SchemaWarning("Constant feature", code="KML-W-S001", severity=WarningSeverity.MEDIUM)

    report = WarningReport(warnings=[w1, w2])

    assert report.total_count == 2
    assert report.by_category == {"Dataset": 1, "Schema": 1}
    assert report.by_severity == {"HIGH": 1, "MEDIUM": 1}
    assert "High missing values" in report.summary_text()
    assert "Training Continued Successfully" in report.summary_text()
