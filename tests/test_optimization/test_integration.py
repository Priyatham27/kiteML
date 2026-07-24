"""
test_integration.py — Integration tests for OptimizationEngine (Story 5.3).
"""

import numpy as np
import pandas as pd
import pytest

from kiteml.optimization import OptimizationEngine


def test_optimization_engine_regression():
    df = pd.DataFrame(
        {
            "x1": np.linspace(1, 50, 20),
            "x2": np.linspace(10, 100, 20),
            "target": np.linspace(5, 50, 20) + np.random.normal(0, 0.5, 20),
        }
    )

    engine = OptimizationEngine()
    res = engine.optimize(
        model_name="RandomForestRegressor",
        dataframe=df,
        target="target",
        problem_type="regression",
        max_trials=2,
    )

    assert res.session.status == "COMPLETED"
    assert res.best_trial is not None
