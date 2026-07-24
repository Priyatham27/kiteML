"""
test_deserializer.py — Unit tests for PipelineDeserializer (Story 4.5).
"""

from pathlib import Path

import pandas as pd
import pytest

from kiteml.pipeline import TransformationPipeline
from kiteml.serialization import PipelineDeserializer, PipelineSerializer


def test_pipeline_deserializer_restores_pipeline(tmp_path: Path):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10, 20, 30]})
    pipeline = TransformationPipeline()
    pipeline.fit(df)

    saved_path = PipelineSerializer().serialize(pipeline, tmp_path / "model.kml")
    restored = PipelineDeserializer().deserialize(saved_path)

    assert isinstance(restored, TransformationPipeline)
    assert restored._is_fitted is True
