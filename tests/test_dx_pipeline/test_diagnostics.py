"""
test_diagnostics.py — Unit tests for Diagnostics and DiagnosticsManager (Story 3.6).
"""

import pytest

from kiteml.pipeline.diagnostics import Diagnostics, DiagnosticsManager


def test_diagnostics():
    diag = Diagnostics(
        status="SUCCESS",
        warning_count=2,
        suggestion_count=4,
        execution_time=1.23,
    )

    summary = diag.summary_text()
    assert "KiteML Diagnostics" in summary
    assert "SUCCESS" in summary
    assert "1.23 sec" in summary


def test_diagnostics_manager():
    mgr = DiagnosticsManager()
    diag = mgr.create_diagnostics(status="WARNINGS", warning_count=1)

    assert diag.status == "WARNINGS"
    assert diag.warning_count == 1
