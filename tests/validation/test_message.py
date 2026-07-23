"""Unit tests for ValidationMessage."""

from kiteml.validation.message import ValidationMessage
from kiteml.validation.severity import ValidationSeverity


def test_validation_message_creation():
    msg = ValidationMessage(
        severity=ValidationSeverity.ERROR,
        title="Target Column Missing",
        description="Target column 'price' was not found in dataset.",
        suggestion="Specify an existing column name.",
        rule_id="R001",
        code="KML1002",
        context={"column": "price"},
    )

    assert msg.severity == ValidationSeverity.ERROR
    assert msg.title == "Target Column Missing"
    assert msg.rule_id == "R001"
    assert msg.code == "KML1002"
    assert msg.context == {"column": "price"}


def test_validation_message_to_dict():
    msg = ValidationMessage(
        severity=ValidationSeverity.WARNING,
        title="High Missing Rate",
        description="30% missing values detected.",
    )
    d = msg.to_dict()

    assert d["severity"] == "warning"
    assert d["title"] == "High Missing Rate"
    assert d["description"] == "30% missing values detected."
    assert d["suggestion"] is None
