"""
loader.py — PackageLoader for restoring model artifacts from .kiteml archives in KiteML.
"""

import json
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kiteml.deployment.validator import PackageValidator


@dataclass
class LoadedPackage:
    """
    Container holding restored model, pipeline, manifest, and descriptor.
    """

    model: Any
    pipeline: Any | None
    manifest: dict[str, Any]
    descriptor: dict[str, Any]

    def predict(self, X: Any) -> Any:
        """Run prediction using restored pipeline and model."""
        from kiteml.prediction.engine import PredictionEngine

        engine = PredictionEngine()
        if hasattr(X, "columns"):
            res = engine.predict(model=self.model, dataframe=X, pipeline=self.pipeline)
            return res.predictions
        return self.model.predict(X)


class PackageLoader:
    """
    Loads and extracts restored model packages from .kiteml archive files.
    """

    def __init__(self) -> None:
        self.validator = PackageValidator()

    def load_package(self, package_path: Path | str) -> LoadedPackage:
        """
        Load and validate .kiteml archive package.

        Parameters
        ----------
        package_path : Path | str
            Path to .kiteml package.

        Returns
        -------
        LoadedPackage
            Restored package container ready for inference.
        """
        p = Path(package_path)
        self.validator.validate_package(p)

        with zipfile.ZipFile(p, "r") as zf:
            names = set(zf.namelist())
            model_bytes = zf.read("model.pkl")
            model = pickle.loads(model_bytes)

            pipeline = None
            if "pipeline.pkl" in names:
                pipeline_bytes = zf.read("pipeline.pkl")
                pipeline = pickle.loads(pipeline_bytes)

            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            descriptor = json.loads(zf.read("descriptor.json").decode("utf-8")) if "descriptor.json" in names else {}

        return LoadedPackage(
            model=model,
            pipeline=pipeline,
            manifest=manifest,
            descriptor=descriptor,
        )
