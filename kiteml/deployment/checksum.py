"""
checksum.py — ChecksumVerifier for computing and verifying SHA-256 package hashes in KiteML.
"""

import hashlib
from pathlib import Path


class ChecksumVerifier:
    """
    Computes SHA-256 integrity hashes for byte buffers and files.
    """

    def compute_sha256(self, data: bytes) -> str:
        """Compute SHA-256 hex digest of bytes."""
        return hashlib.sha256(data).hexdigest()

    def compute_file_sha256(self, file_path: Path | str) -> str:
        """Compute SHA-256 hex digest of file on disk."""
        path = Path(file_path)
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        return sha.hexdigest()

    def verify(self, data: bytes, expected_hash: str) -> bool:
        """Verify bytes against expected SHA-256 hash."""
        return self.compute_sha256(data) == expected_hash
