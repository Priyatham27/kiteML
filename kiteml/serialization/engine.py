"""
engine.py — Public pipeline save and load API functions for KiteML.
"""

from pathlib import Path
from typing import Any

from kiteml.serialization.deserializer import PipelineDeserializer
from kiteml.serialization.serializer import PipelineSerializer


def save_pipeline(pipeline: Any, filepath: str | Path) -> str:
    """
    Save fitted TransformationPipeline to a native .kml archive package file.

    Parameters
    ----------
    pipeline : TransformationPipeline
        Fitted transformation pipeline.
    filepath : str | Path
        Destination .kml filepath.

    Returns
    -------
    str
        Absolute saved filepath.
    """
    serializer = PipelineSerializer()
    return serializer.serialize(pipeline, filepath)


def load_pipeline(filepath: str | Path) -> Any:
    """
    Load fitted TransformationPipeline from a native .kml archive package file.

    Parameters
    ----------
    filepath : str | Path
        Path to .kml package file.

    Returns
    -------
    TransformationPipeline
        Restored pipeline instance ready for inference transformations.
    """
    deserializer = PipelineDeserializer()
    return deserializer.deserialize(filepath)
