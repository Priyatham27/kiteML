"""
validator.py — SerializationValidator integrity checker for KiteML .kml packages.
"""

import zipfile
from typing import Any

from kiteml.serialization.checksum import ChecksumManager
from kiteml.serialization.manifest import PipelineManifest


class SerializationValidator:
    """
    Validates structural completeness and SHA-256 integrity of .kml archive packages.
    """

    REQUIRED_FILES = ("manifest.json", "metadata.json", "pipeline.pkl", "checksum.sha256")

    def validate_archive(self, zip_ref: zipfile.ZipFile) -> tuple[bool, list[str]]:
        """
        Validate .kml zip archive contents and SHA-256 checksum.

        Returns
        -------
        tuple[bool, list[str]]
            (is_valid, list_of_errors)
        """
        errors: list[str] = []
        namelist = zip_ref.namelist()

        for req in self.REQUIRED_FILES:
            if req not in namelist:
                errors.append(f"Invalid .kml package: Missing required file '{req}'.")

        if errors:
            return (False, errors)

        pipeline_bytes = zip_ref.read("pipeline.pkl")
        expected_hash = zip_ref.read("checksum.sha256").decode("utf-8").strip()

        if not ChecksumManager.verify_bytes_hash(pipeline_bytes, expected_hash):
            errors.append(
                "Package integrity failure: SHA-256 checksum mismatch (file may be corrupted or tampered with)."
            )

        return (len(errors) == 0, errors)
