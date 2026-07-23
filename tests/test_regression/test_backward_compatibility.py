"""
test_backward_compatibility.py — Regression tests ensuring public API backward compatibility (Story 3.7).
"""

import pandas as pd
import pytest

import kiteml


def test_train_backward_compatibility():
    df = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 22, 28, 33, 38, 43, 48, 53, 58, 63, 68],
            "income": [
                50000,
                60000,
                70000,
                80000,
                90000,
                100000,
                110000,
                120000,
                130000,
                140000,
                45000,
                55000,
                65000,
                75000,
                85000,
                95000,
                105000,
                115000,
                125000,
                135000,
            ],
            "bought": [0, 1] * 10,
        }
    )

    res = kiteml.train(df, target="bought", validate_data=False)

    assert res.model is not None
    assert res.model_name is not None
    assert isinstance(res.metrics, dict) or hasattr(res.metrics, "to_dict")
    assert isinstance(res.report_text, str)
    assert hasattr(res, "warning_summary")
    assert hasattr(res, "suggestions")
    assert hasattr(res, "diagnostics")
