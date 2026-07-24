"""
test_stages.py — Unit tests for individual PipelineStage classes (Story 4.4).
"""

import numpy as np
import pandas as pd
import pytest

from kiteml.pipeline import (
    DatetimeStage,
    EncodingStage,
    MissingValueStage,
    ScalingStage,
)


def test_missing_value_stage():
    df = pd.DataFrame(
        {
            "num": [1.0, np.nan, 3.0],
            "cat": ["A", np.nan, "A"],
        }
    )

    stage = MissingValueStage()
    df_out = stage.fit_transform(df)

    assert df_out["num"].isna().sum() == 0
    assert df_out["cat"].isna().sum() == 0
    assert df_out["num"].iloc[1] == 2.0


def test_datetime_stage():
    df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=3, freq="D")})
    stage = DatetimeStage()
    df_out = stage.fit_transform(df)

    assert "date_year" in df_out.columns
    assert "date_weekday" in df_out.columns
    assert "date" not in df_out.columns


def test_encoding_stage():
    df = pd.DataFrame({"color": ["red", "blue", "red"]})
    stage = EncodingStage()
    df_out = stage.fit_transform(df)

    assert "color_blue" in df_out.columns
    assert "color_red" in df_out.columns


def test_scaling_stage():
    df = pd.DataFrame({"val": [10.0, 20.0, 30.0]})
    stage = ScalingStage()
    df_out = stage.fit_transform(df)

    assert abs(df_out["val"].mean()) < 1e-5
