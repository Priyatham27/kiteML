"""
test_formatter.py — Unit tests for WarningFormatter (Story 3.4).
"""

import json

import pytest

from kiteml.warnings import DatasetWarning, WarningFormatter, WarningReport


def test_warning_formatter():
    w = DatasetWarning("Duplicate rows found", code="KML-W-D002", recommendation="Remove duplicates")
    fmt = WarningFormatter()

    term_out = fmt.to_terminal(w)
    assert "KML-W-D002" in term_out
    assert "Duplicate rows found" in term_out

    json_out = fmt.to_json(w)
    parsed = json.loads(json_out)
    assert parsed["code"] == "KML-W-D002"
    assert parsed["recommendation"] == "Remove duplicates"


def test_warning_report_formatter():
    report = WarningReport(
        warnings=[
            DatasetWarning("High missing values", code="KML-W-D001"),
            DatasetWarning("Duplicate rows", code="KML-W-D002"),
        ]
    )
    fmt = WarningFormatter()

    term_out = fmt.to_terminal(report)
    assert "KiteML Warning Summary" in term_out
    assert "Dataset" in term_out

    json_out = fmt.to_json(report)
    parsed = json.loads(json_out)
    assert parsed["total_count"] == 2
