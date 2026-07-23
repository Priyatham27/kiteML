"""
test_formatter.py — Unit tests for ErrorFormatter (Story 3.2).
"""

import json

import pytest

from kiteml.exceptions import ErrorFormatter, TargetError


def test_error_formatter_methods():
    err = TargetError(
        message='Target column "price" not found.',
        error_code="KML-T001",
        suggestion="Choose one of the available columns.",
        context={"available_columns": ["age", "salary", "city"]},
    )
    formatter = ErrorFormatter()

    term_out = formatter.to_terminal(err)
    assert "KML-T001" in term_out
    assert 'Target column "price" not found.' in term_out

    text_out = formatter.to_text(err)
    assert "[KML-T001]" in text_out

    json_out = formatter.to_json(err)
    parsed = json.loads(json_out)
    assert parsed["error_code"] == "KML-T001"

    dict_out = formatter.to_dict(err)
    assert dict_out["error_code"] == "KML-T001"

    html_out = formatter.to_html(err)
    assert "kiteml-error-container" in html_out

    md_out = formatter.to_markdown(err)
    assert "> [!CAUTION]" in md_out


def test_formatter_mode_parameter():
    err = TargetError(message="Missing target")
    formatter = ErrorFormatter()

    assert "KML-T000" in formatter.format(err, mode="terminal")
    assert "[KML-T000]" in formatter.format(err, mode="text")
    assert '"error_code": "KML-T000"' in formatter.format(err, mode="json")
