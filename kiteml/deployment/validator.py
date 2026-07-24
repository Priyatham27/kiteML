"""
validator.py — PackageValidator for checking .kiteml archive integrity in KiteML.
"""

import json
import zipfile
from pathlib import Path

from kiteml.deployment.checksum import ChecksumVerifier


class PackageValidator:
    """
    Validates zip archive contents and SHA-256 checksums for .kiteml packages.
    """

    def __init__(self) -> None:
        self.checksum_verifier = ChecksumVerifier()

    def validate_package(self, package_path: Path | str) -> bool:
        """
        Validate package integrity and checksum.

        Parameters
        ----------
        package_path : Path | str
            Path to .kiteml package.

        Returns
        -------
        bool
            True if package is valid and intact.
        """
        p = Path(package_path)
        if not p.exists():
            raise FileNotFoundError(f"Package file '{p}' does not exist.")

        if not zipfile.is_zipfile(p):
            raise ValueError(f"File '{p}' is not a valid zip archive package.")

        with zipfile.ZipFile(p, "r") as zf:
            names = set(zf.namelist())
            if "model.pkl" not in names or "manifest.json" not in names:
                raise ValueError("Package is missing required 'model.pkl' or 'manifest.json' files.")

            manifest_bytes = zf.read("manifest.json")
            manifest = json.loads(manifest_bytes.decode("utf-8"))

            expected_checksum = manifest.get("checksum", "")

            model_bytes = zf.read("model.pkl")
            pipeline_bytes = zf.read("pipeline.pkl") if "pipeline.pkl" in names else b""

            actual_checksum = self.checksum_verifier.compute_sha256(model_bytes + pipeline_bytes)

            if expected_checksum and actual_checksum != expected_checksum:
                raise ValueError(
                    f"Package SHA-256 checksum mismatch! Expected '{expected_checksum}', got '{actual_checksum}'."
                )

        return True
