"""
test_context_and_state.py — Unit tests for PipelineContext and PipelineState (Story 4.4).
"""

import pandas as pd
import pytest

from kiteml.pipeline import PipelineContext, PipelineState


def test_pipeline_context_and_state():
    df = pd.DataFrame({"a": [1, 2, 3]})
    ctx = PipelineContext(original_df=df, current_df=df)
    assert ctx.original_df is not None
    state = PipelineState()

    state.mark_completed("MissingValueStage")
    state.mark_skipped("DatetimeStage")

    d = state.to_dict()
    assert d["completed_stages"] == ["MissingValueStage"]
    assert d["skipped_stages"] == ["DatetimeStage"]
    assert d["success"] is True
