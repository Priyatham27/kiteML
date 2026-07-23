"""
test_performance.py — Performance benchmark regression tests (Story 3.7).
"""

import time

import pandas as pd
import pytest

from kiteml.exceptions import ErrorFormatter, TargetError
from kiteml.pipeline import ContextBuilder, DiagnosticsManager
from kiteml.suggestions import SuggestionEngine
from kiteml.warnings import DatasetWarning, WarningCollector


def test_performance_error_formatter():
    err = TargetError("Target column 'price' not found", error_code="KML-T002")
    fmt = ErrorFormatter()

    start = time.perf_counter()
    _ = fmt.format(err, mode="terminal")
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # Target < 2 ms
    assert elapsed_ms < 5.0  # Safe upper threshold across platforms


def test_performance_suggestion_engine():
    engine = SuggestionEngine()
    df = pd.DataFrame({"Price": range(100), "Age": range(100)})

    start = time.perf_counter()
    _ = engine.generate({"df": df, "target": "prcie", "available_columns": ["Price", "Age"]})
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # Target < 10 ms
    assert elapsed_ms < 15.0


def test_performance_warning_collector():
    collector = WarningCollector()
    start = time.perf_counter()
    for i in range(1000):
        collector.add(DatasetWarning(f"Warning {i}", code=f"KML-W-D{i:03d}"))
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # Threshold < 50 ms for 1000 warnings across environments
    assert elapsed_ms < 50.0


def test_performance_diagnostics():
    mgr = DiagnosticsManager()
    start = time.perf_counter()
    diag = mgr.create_diagnostics(status="SUCCESS", warning_count=5, suggestion_count=10, execution_time=1.5)
    _ = diag.summary_text()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # Target < 3 ms
    assert elapsed_ms < 5.0
