"""
test_integration.py — Integration tests for TrainingEngine (Story 5.1).
"""

import numpy as np
import pandas as pd
import pytest

from kiteml.training import TrainingEngine


def test_training_engine_regression_task():
    df = pd.DataFrame(
        {
            "x1": np.linspace(1, 100, 20),
            "x2": np.linspace(50, 150, 20),
            "target": np.linspace(10, 200, 20) + np.random.normal(0, 1, 20),
        }
    )

    engine = TrainingEngine()
    result = engine.train(dataframe=df, target="target", problem_type="regression")

    assert result.session.task_type == "regression"
    assert result.fitted_model is not None
