"""
test_context_builder.py — Unit tests for ContextBuilder (Story 3.6).
"""

import pandas as pd
import pytest

from kiteml.pipeline.context_builder import ContextBuilder


def test_context_builder_build():
    builder = ContextBuilder()
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    ctx = builder.build_context(df=df, target="b", pipeline_stage="training", model_name="RandomForest")

    assert ctx["dataset_rows"] == 2
    assert ctx["dataset_columns"] == 2
    assert ctx["available_columns"] == ["a", "b"]
    assert ctx["target"] == "b"
    assert ctx["pipeline_stage"] == "training"
    assert ctx["model_name"] == "RandomForest"
