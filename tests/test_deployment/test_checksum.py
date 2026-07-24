"""
test_checksum.py — Unit tests for ChecksumVerifier (Story 5.7).
"""

import pytest

from kiteml.deployment import ChecksumVerifier


def test_checksum_verifier():
    verifier = ChecksumVerifier()
    data = b"hello kiteml deployment"

    sha = verifier.compute_sha256(data)
    assert len(sha) == 64
    assert verifier.verify(data, sha) is True
    assert verifier.verify(b"different data", sha) is False
