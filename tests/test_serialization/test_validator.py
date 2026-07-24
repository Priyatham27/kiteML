"""
test_validator.py — Unit tests for SerializationValidator (Story 4.5).
"""

import zipfile
from pathlib import Path

import pytest

from kiteml.serialization import SerializationValidator


def test_validator_detects_missing_file(tmp_path: Path):
    zip_path = tmp_path / "test.kml"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("manifest.json", "{}")

    validator = SerializationValidator()
    with zipfile.ZipFile(zip_path, "r") as zf:
        valid, errors = validator.validate_archive(zf)

    assert valid is False
    assert any("Missing required file" in err for err in errors)
