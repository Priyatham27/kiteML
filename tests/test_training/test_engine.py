"""
test_engine.py — Unit tests for TrainingEngine (Story 5.1).
"""

import pandas as pd
import pytest

from kiteml.training import TrainingEngine, TrainingResult


def test_training_engine_classification():
    df = pd.DataFrame(
        {
            "price": [10.0, 20.0, 30.0, 40.0, 50.0] * 4,
            "qty": [1, 2, 3, 4, 5] * 4,
            "target": [0, 1] * 10,
        }
    )

    engine = TrainingEngine()
    result = engine.train(df, target="target")

    assert isinstance(result, TrainingResult)
    assert result.session.status == "COMPLETED"
    assert result.fitted_model is not None
    assert "Training Session Result" in result.summary()
