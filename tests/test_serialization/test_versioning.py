"""
test_versioning.py — Unit tests for VersionManager (Story 4.5).
"""

import sys

import pytest

from kiteml.serialization import PipelineManifest, VersionManager


def test_version_manager_compatibility():
    manifest = PipelineManifest(python_version=sys.version.split()[0])
    is_compat, msg = VersionManager.check_compatibility(manifest)

    assert is_compat is True
    assert "fully compatible" in msg
