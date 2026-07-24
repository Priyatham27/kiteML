"""
test_pipeline.py — Unit tests for PipelineReplayEngine (Story 5.6).
"""

import pandas as pd
import pytest

from kiteml.prediction import PipelineReplayEngine


class MockPipeline:

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["c"] = df["a"] + df["b"]
        return df


def test_pipeline_replay():
    engine = PipelineReplayEngine()
    df = pd.DataFrame({"a": [1, 2], "b": [10, 20]})

    transformed = engine.replay(MockPipeline(), df)
    assert "c" in transformed.columns
    assert transformed.iloc[0]["c"] == 11
