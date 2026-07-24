"""
test_integration.py — Integration tests for PredictionEngine (Story 5.6).
"""

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from kiteml.prediction import PredictionEngine


def test_prediction_engine_batch_and_streaming():
    X = pd.DataFrame({"x": range(20)})
    y = pd.Series(range(20))

    reg = LinearRegression()
    reg.fit(X, y)

    engine = PredictionEngine()

    batch_res = engine.batch_predict(model=reg, dataframe=X, chunk_size=5)
    assert len(batch_res) == 20

    stream_res = engine.stream_predict(model=reg, records_iterator=[{"x": 1}, {"x": 2}])
    assert len(stream_res) == 2
