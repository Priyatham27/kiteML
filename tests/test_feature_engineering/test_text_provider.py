"""
test_text_provider.py — Unit tests for TextFEProvider (Story 4.2).
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from kiteml.feature_engineering import FeatureImportancePredictor, FERuleEngine, TextFEProvider


def test_text_fe_provider():
    df = pd.DataFrame({"review": ["Great product!", "Not bad", "Terrible quality"] * 5})
    provider = TextFEProvider()
    rules = FERuleEngine()
    predictor = FeatureImportancePredictor()

    mock_profile = MagicMock()
    mock_profile.text.text_columns = ["review"]

    plans = provider.plan_features(df, mock_profile, rules, predictor)

    assert len(plans) == 2
    gen_names = [p.generated_name for p in plans]
    assert "review_word_count" in gen_names
    assert "review_char_count" in gen_names
