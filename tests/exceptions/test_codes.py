"""
test_codes.py — Unit tests for error code constants (Story 3.3).
"""

import re

import pytest

from kiteml.exceptions import codes


def test_error_codes_format_and_uniqueness():
    code_attrs = [getattr(codes, name) for name in dir(codes) if name.startswith("KML_")]

    assert len(code_attrs) > 0

    # Ensure uniqueness across all defined error codes
    unique_codes = set(code_attrs)
    assert len(unique_codes) == len(code_attrs)

    # Ensure KML-XXXNNN pattern format
    pattern = re.compile(r"^KML-[A-Z]+[0-9]+$")
    for code in code_attrs:
        assert pattern.match(code), f"Invalid error code format: {code}"
