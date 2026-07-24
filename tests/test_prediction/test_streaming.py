"""
test_streaming.py — Unit tests for StreamPredictor (Story 5.6).
"""

import pandas as pd
import pytest

from kiteml.prediction import StreamPredictor


def test_stream_predictor():
    records = [{"a": 1}, {"a": 2}, {"a": 3}]

    def dummy_predict(df_single: pd.DataFrame):
        return [df_single.iloc[0]["a"] * 10]

    stream_preds = StreamPredictor().stream_predict(dummy_predict, records)
    assert stream_preds == [10, 20, 30]
