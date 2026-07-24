"""
deserializer.py — PipelineDeserializer native .kml package reconstructor for KiteML.
"""

import pickle
import zipfile
from pathlib import Path
from typing import Any

from kiteml.serialization.manifest import PipelineManifest
from kiteml.serialization.validator import SerializationValidator
from kiteml.serialization.versioning import VersionManager


class PipelineDeserializer:
    """
    Deserializes and reconstructs a fitted TransformationPipeline from a .kml archive package.
    """

    def __init__(self) -> None:
        self.validator = SerializationValidator()

    def deserialize(self, filepath: str | Path) -> Any:
        """
        Unpack and restore TransformationPipeline from a .kml archive package.

        Parameters
        ----------
        filepath : str | Path
            Path to .kml package file.

        Returns
        -------
        TransformationPipeline
            Restored transformation pipeline.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Serialized pipeline package not found: {path}")

        with zipfile.ZipFile(path, "r") as zip_ref:
            valid, errors = self.validator.validate_archive(zip_ref)
            if not valid:
                raise ValueError(f"Corrupted or invalid .kml pipeline package: {'; '.join(errors)}")

            manifest_bytes = zip_ref.read("manifest.json")
            manifest = PipelineManifest.from_json(manifest_bytes.decode("utf-8"))

            is_compat, msg = VersionManager.check_compatibility(manifest)
            if not is_compat:
                raise RuntimeError(f"Incompatible .kml package version: {msg}")

            pipeline_bytes = zip_ref.read("pipeline.pkl")
            pipeline = pickle.loads(pipeline_bytes)
            return pipeline
