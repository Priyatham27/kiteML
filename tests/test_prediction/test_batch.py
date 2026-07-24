"""
test_batch.py — Unit tests for BatchPredictor (Story 5.6).
"""

import pandas as pd
import pytest

from kiteml.prediction import BatchPredictor


def test_batch_predictor():
    df = pd.DataFrame({"a": range(10)})

    def dummy_predict(chunk: pd.DataFrame):
        return chunk["a"] * 2

    preds = BatchPredictor().batch_predict(dummy_predict, df, chunk_size=3)
    assert len(preds) == 10
    assert preds[0] == 0
    assert preds[9] == 18
