"""
test_serializer.py — Unit tests for PipelineSerializer (Story 4.5).
"""

from pathlib import Path

import pandas as pd
import pytest

from kiteml.pipeline import TransformationPipeline
from kiteml.serialization import PipelineSerializer


def test_pipeline_serializer_creates_kml(tmp_path: Path):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10, 20, 30]})
    pipeline = TransformationPipeline()
    pipeline.fit(df)

    serializer = PipelineSerializer()
    saved_path = serializer.serialize(pipeline, tmp_path / "model.kml")

    assert Path(saved_path).exists()
    assert Path(saved_path).suffix == ".kml"
