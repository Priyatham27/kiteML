"""
builder.py — PackageBuilder for bundling model artifacts into .kiteml zip archives in KiteML.
"""

import json
import pickle
import zipfile
from pathlib import Path
from typing import Any

from kiteml.deployment.checksum import ChecksumVerifier
from kiteml.deployment.descriptor import UniversalDeploymentDescriptor
from kiteml.deployment.manifest import DeploymentManifest


class PackageBuilder:
    """
    Bundles model, pipeline, manifest, descriptor, and SHA-256 checksum into a portable .kiteml zip package.
    """

    def __init__(self) -> None:
        self.checksum_verifier = ChecksumVerifier()

    def build_package(
        self,
        model: Any,
        model_name: str,
        task_type: str,
        output_path: Path | str,
        pipeline: Any | None = None,
        feature_names: list[str] | None = None,
    ) -> Path:
        """
        Build .kiteml package file.

        Parameters
        ----------
        model : Any
            Fitted model estimator.
        model_name : str
            Model algorithm name.
        task_type : str
            ML task type.
        output_path : Path | str
            Destination file path for .kiteml archive.
        pipeline : Any, optional
            Fitted transformation pipeline.
        feature_names : list[str], optional
            List of feature names.

        Returns
        -------
        Path
            Path to generated .kiteml archive.
        """
        out = Path(output_path)
        if not out.name.endswith(".kiteml") and not out.name.endswith(".zip"):
            out = out.with_suffix(".kiteml")

        out.parent.mkdir(parents=True, exist_ok=True)

        model_bytes = pickle.dumps(model)
        pipeline_bytes = pickle.dumps(pipeline) if pipeline is not None else b""

        raw_checksum = self.checksum_verifier.compute_sha256(model_bytes + pipeline_bytes)

        manifest = DeploymentManifest(
            model_name=model_name,
            task_type=task_type,
            checksum=raw_checksum,
        )

        descriptor = UniversalDeploymentDescriptor(
            model_name=model_name,
            task_type=task_type,
            feature_names=feature_names or [],
        )

        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("model.pkl", model_bytes)
            if pipeline is not None:
                zf.writestr("pipeline.pkl", pipeline_bytes)
            zf.writestr("manifest.json", json.dumps(manifest.to_dict(), indent=2))
            zf.writestr("descriptor.json", json.dumps(descriptor.to_dict(), indent=2))
            zf.writestr("checksum.sha256", raw_checksum)

        return out
