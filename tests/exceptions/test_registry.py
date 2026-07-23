"""
test_registry.py — Unit tests for ErrorRegistry (Story 3.3).
"""

import pytest

from kiteml.exceptions import ErrorDefinition, ErrorRegistry, global_error_registry


def test_registry_integrity():
    assert global_error_registry.validate_integrity() is True
    assert len(global_error_registry.all_definitions()) >= 50


def test_custom_registration():
    reg = ErrorRegistry()
    custom_def = ErrorDefinition(
        code="KML-TEST01",
        name="Test Custom Error",
        category="Testing",
        message_template="Custom test message: {arg}",
        default_suggestion="Fix custom test",
    )
    reg.register(custom_def)

    assert reg.contains("KML-TEST01")
    retrieved = reg.get("KML-TEST01")
    assert retrieved is custom_def
    assert retrieved.format_message(arg="sample") == "Custom test message: sample"
