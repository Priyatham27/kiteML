"""
test_numeric_provider.py — Unit tests for NumericFEProvider (Story 4.2).
"""

import pandas as pd
import pytest

from kiteml.feature_engineering import FeatureImportancePredictor, FERuleEngine, NumericFEProvider


def test_numeric_fe_provider():
    # Right-skewed positive data
    df = pd.DataFrame({"income": [10.0, 10.0, 15.0, 20.0, 100.0, 500.0, 1000.0] * 3})
    provider = NumericFEProvider()
    rules = FERuleEngine(skewness_threshold=1.0)
    predictor = FeatureImportancePredictor()

    plans = provider.plan_features(df, None, rules, predictor)

    assert len(plans) == 1
    assert plans[0].generated_name == "log_income"
