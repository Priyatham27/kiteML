"""
Tests for kiteml.preprocessing modules.
"""

import pytest
import pandas as pd
import numpy as np

from kiteml.preprocessing.cleaner import handle_missing_values
from kiteml.preprocessing.encoder import encode_categoricals
from kiteml.preprocessing.scaler import scale_features


class TestCleaner:
    """Tests for handle_missing_values."""

    def test_fill_numeric_median(self):
        df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [10, np.nan, 30, 40]})
        cleaned = handle_missing_values(df, strategy="median")
        assert cleaned.isnull().sum().sum() == 0

    def test_fill_categorical_mode(self):
        df = pd.DataFrame({"cat": ["a", "b", None, "a"]})
        cleaned = handle_missing_values(df, strategy="auto")
        assert cleaned.isnull().sum().sum() == 0
        assert cleaned["cat"].iloc[2] == "a"

    def test_drop_strategy(self):
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, 6]})
        cleaned = handle_missing_values(df, strategy="drop")
        assert len(cleaned) == 2


class TestEncoder:
    """Tests for encode_categoricals."""

    def test_encode_object_columns(self):
        df = pd.DataFrame({"color": ["red", "blue", "green"], "target": [0, 1, 0]})
        encoded, encoders = encode_categoricals(df, target="target")
        assert encoded["color"].dtype in [np.int32, np.int64]
        assert "color" in encoders

    def test_no_categorical_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        encoded, encoders = encode_categoricals(df, target="b")
        assert len(encoders) == 0


class TestScaler:
    """Tests for scale_features."""

    def test_scale_features(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        scaled, scaler = scale_features(df)
        assert abs(scaled["a"].mean()) < 1e-10
        assert scaler is not None

    def test_reuse_scaler(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        _, scaler = scale_features(df)
        new_df = pd.DataFrame({"a": [7, 8], "b": [9, 10]})
        scaled, _ = scale_features(new_df, scaler=scaler)
        assert scaled.shape == (2, 2)
