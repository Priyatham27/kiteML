"""
test_integration.py — End-to-end integration tests for Story 5.8 in KiteML.
"""

import pandas as pd
import pytest

from kiteml import train


def test_end_to_end_regression():
    # Use 20 rows so StratifiedKFold n_splits=5 doesn't fail
    n = 20
    df = pd.DataFrame(
        {
            "x1": [float(i) for i in range(n)],
            "x2": [float(i * 10) for i in range(n)],
            "target": [float(15 + i * 10) for i in range(n)],
        }
    )

    result = train(dataframe=df, target="target", validate_data=False, problem_type="regression")
    assert result.problem_type in ("regression", "continuous")
    assert result.model is not None

    preds = result.predict(df.drop(columns=["target"]))
    assert len(preds) == n
