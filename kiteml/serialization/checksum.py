"""
checksum.py — SHA-256 checksum manager for KiteML pipeline serialization integrity.
"""

import hashlib
from pathlib import Path


class ChecksumManager:
    """
    Computes and verifies SHA-256 hashes for serialized pipeline components.
    """

    @staticmethod
    def compute_bytes_hash(data: bytes) -> str:
        """Compute SHA-256 hex digest of raw byte content."""
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def compute_file_hash(filepath: str | Path) -> str:
        """Compute SHA-256 hex digest of a file on disk."""
        sha = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                sha.update(chunk)
        return sha.hexdigest()

    @staticmethod
    def verify_bytes_hash(data: bytes, expected_hash: str) -> bool:
        """Verify that data SHA-256 matches expected hash."""
        return ChecksumManager.compute_bytes_hash(data) == expected_hash
