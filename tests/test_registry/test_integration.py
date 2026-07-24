"""
test_integration.py — Integration tests for TrainingEngine + ModelRegistry (Story 5.2).
"""

import pandas as pd
import pytest

from kiteml.training import TrainingEngine


def test_training_engine_uses_model_registry():
    df = pd.DataFrame(
        {
            "price": [10.0, 20.0, 30.0, 40.0, 50.0] * 4,
            "qty": [1, 2, 3, 4, 5] * 4,
            "target": [0, 1] * 10,
        }
    )

    engine = TrainingEngine()
    result = engine.train(df, target="target")

    assert result.fitted_model is not None
    assert result.session.status == "COMPLETED"
