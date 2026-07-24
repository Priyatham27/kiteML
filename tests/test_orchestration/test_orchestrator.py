"""
test_orchestrator.py — Unit tests for KiteMLPipeline (Story 4.7).
"""

from pathlib import Path

import pandas as pd
import pytest

from kiteml import KiteMLPipeline, PipelineBuildResult


def test_kiteml_pipeline_build():
    df = pd.DataFrame(
        {
            "price": [10.0, 20.0, 30.0, 40.0],
            "qty": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )

    pipeline = KiteMLPipeline()
    result = pipeline.build(df, target="target")

    assert isinstance(result, PipelineBuildResult)
    assert not result.transformed_df.empty
    assert result.report is not None


def test_kiteml_pipeline_save_and_load(tmp_path: Path):
    df = pd.DataFrame(
        {
            "price": [10.0, 20.0, 30.0, 40.0],
            "qty": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )

    pipeline = KiteMLPipeline()
    pipeline.build(df, target="target")

    save_path = tmp_path / "model.kml"
    saved = pipeline.save(save_path)

    assert Path(saved).exists()
    loaded_pipeline = KiteMLPipeline.load(saved)
    assert loaded_pipeline.pipeline is not None
