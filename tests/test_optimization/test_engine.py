"""
test_engine.py — Unit tests for OptimizationEngine (Story 5.3).
"""

import pandas as pd
import pytest

from kiteml.optimization import OptimizationEngine, OptimizationResult


def test_optimization_engine_run():
    df = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0] * 4,
            "x2": [10, 20, 30, 40, 50] * 4,
            "target": [0, 1] * 10,
        }
    )

    engine = OptimizationEngine()
    result = engine.optimize(
        model_name="RandomForestClassifier",
        dataframe=df,
        target="target",
        max_trials=3,
    )

    assert isinstance(result, OptimizationResult)
    assert len(result.trials) > 0
    assert result.best_score > 0.0
