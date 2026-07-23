"""
test_error_pipeline.py — Unit tests for ErrorManager (Story 3.6).
"""

import pytest

from kiteml.exceptions import TargetError
from kiteml.pipeline.error_manager import ErrorManager


def test_error_manager_processing():
    mgr = ErrorManager()
    err = ValueError("Raw value error")

    processed = mgr.process_error(err)
    assert hasattr(processed, "error_code")

    formatted = mgr.format_error(TargetError("Target missing", error_code="KML-T001"))
    assert "KML-T001" in formatted
