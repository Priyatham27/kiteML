"""
test_datetime_provider.py — Unit tests for DatetimeFEProvider (Story 4.2).
"""

import pandas as pd
import pytest

from kiteml.feature_engineering import DatetimeFEProvider, FeatureImportancePredictor, FERuleEngine


def test_datetime_fe_provider():
    df = pd.DataFrame({"purchase_date": pd.date_range("2025-01-01", periods=10, freq="D")})
    provider = DatetimeFEProvider()
    rules = FERuleEngine()
    predictor = FeatureImportancePredictor()

    plans = provider.plan_features(df, None, rules, predictor)

    assert len(plans) == 6
    gen_names = [p.generated_name for p in plans]
    assert "purchase_date_year" in gen_names
    assert "purchase_date_weekday" in gen_names
    assert "purchase_date_is_weekend" in gen_names
