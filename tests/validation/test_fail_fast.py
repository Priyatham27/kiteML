"""
test_fail_fast.py — Unit tests for fail-fast behavior in ValidationPipeline (Story 2.6).
"""

import pandas as pd
import pytest

from kiteml.validation.pipeline import ValidationPipeline


def test_fail_fast_on_none_dataset():
    pipeline = ValidationPipeline()
    summary = pipeline.validate(None, target="target", fail_fast=True)

    assert summary.passed is False
    assert summary.ready_for_training is False
    # TargetValidator, SchemaValidator, QualityValidator should be skipped
    assert "DatasetValidator" in summary.validator_results
    assert "TargetValidator" not in summary.validator_results


def test_fail_fast_on_empty_dataframe():
    pipeline = ValidationPipeline()
    df_empty = pd.DataFrame()
    summary = pipeline.validate(df_empty, target="target", fail_fast=True)

    assert summary.passed is False
    assert summary.ready_for_training is False
    assert "DatasetValidator" in summary.validator_results
    assert "TargetValidator" not in summary.validator_results


def test_fail_fast_on_missing_target():
    pipeline = ValidationPipeline()
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    summary = pipeline.validate(df, target="non_existent_target", fail_fast=True)

    assert summary.passed is False
    assert summary.ready_for_training is False
    assert "DatasetValidator" in summary.validator_results
    assert "TargetValidator" in summary.validator_results
    assert "SchemaValidator" not in summary.validator_results
