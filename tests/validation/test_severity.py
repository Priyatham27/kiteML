"""Unit tests for ValidationSeverity enum."""

from kiteml.validation.severity import ValidationSeverity


def test_severity_ordering():
    assert ValidationSeverity.INFO < ValidationSeverity.WARNING
    assert ValidationSeverity.WARNING < ValidationSeverity.ERROR
    assert ValidationSeverity.ERROR < ValidationSeverity.CRITICAL


def test_severity_string_comparison():
    assert ValidationSeverity.INFO == "info"
    assert ValidationSeverity.WARNING == "WARNING"
    assert ValidationSeverity.ERROR > "warning"
    assert ValidationSeverity.CRITICAL > "error"


def test_severity_ranks():
    assert ValidationSeverity.INFO.rank == 1
    assert ValidationSeverity.WARNING.rank == 2
    assert ValidationSeverity.ERROR.rank == 3
    assert ValidationSeverity.CRITICAL.rank == 4
