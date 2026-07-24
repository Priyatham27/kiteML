"""
test_checksum.py — Unit tests for ChecksumManager (Story 4.5).
"""

import pytest

from kiteml.serialization import ChecksumManager


def test_checksum_manager_sha256():
    data = b"KiteML Pipeline Content"
    hash_val = ChecksumManager.compute_bytes_hash(data)

    assert isinstance(hash_val, str)
    assert len(hash_val) == 64
    assert ChecksumManager.verify_bytes_hash(data, hash_val) is True
    assert ChecksumManager.verify_bytes_hash(b"Tampered Content", hash_val) is False
