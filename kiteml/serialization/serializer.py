"""
serializer.py — PipelineSerializer native .kml package builder for KiteML.
"""

import json
import pickle
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from kiteml.serialization.checksum import ChecksumManager
from kiteml.serialization.manifest import PipelineManifest


class PipelineSerializer:
    """
    Serializes a fitted TransformationPipeline into a structured, versioned .kml archive package.
    """

    def serialize(self, pipeline: Any, filepath: str | Path) -> str:
        """
        Serialize transformation pipeline into a native .kml zip archive package.

        Parameters
        ----------
        pipeline : TransformationPipeline
            Fitted transformation pipeline.
        filepath : str | Path
            Target destination path for .kml package.

        Returns
        -------
        str
            Absolute path to saved .kml package.
        """
        target_path = Path(filepath)
        if target_path.suffix != ".kml":
            target_path = target_path.with_suffix(".kml")

        target_path.parent.mkdir(parents=True, exist_ok=True)

        pipeline_bytes = pickle.dumps(pipeline)
        checksum = ChecksumManager.compute_bytes_hash(pipeline_bytes)

        stage_names = [s.name for s in getattr(pipeline, "fitted_stages", [])]
        target_name = getattr(pipeline.context, "target_name", None) if hasattr(pipeline, "context") else None
        feat_count = (
            len(getattr(pipeline.context, "current_df", pd.DataFrame()).columns)
            if hasattr(pipeline, "context") and pipeline.context.current_df is not None
            else 0
        )

        manifest = PipelineManifest(
            checksum=checksum,
            feature_count=feat_count,
            target_name=target_name,
            stage_names=stage_names,
        )

        metadata = {
            "target_name": target_name,
            "problem_type": getattr(pipeline.context, "problem_type", None) if hasattr(pipeline, "context") else None,
            "feature_count": feat_count,
            "stage_names": stage_names,
        }

        with zipfile.ZipFile(target_path, "w", zipfile.ZIP_DEFLATED) as zip_out:
            zip_out.writestr("manifest.json", manifest.to_json())
            zip_out.writestr("metadata.json", json.dumps(metadata, indent=2))
            zip_out.writestr("pipeline.pkl", pipeline_bytes)
            zip_out.writestr("checksum.sha256", checksum)

        return str(target_path.resolve())
