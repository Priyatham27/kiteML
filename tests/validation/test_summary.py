"""
test_summary.py — Unit tests for ValidationSummary (Story 2.6).
"""

import json

import pandas as pd
import pytest

from kiteml.validation.pipeline import ValidationPipeline
from kiteml.validation.validation_summary import ValidationSummary


def test_summary_properties_and_serialization():
    summary = ValidationSummary(
        passed=True,
        health_score=94,
        health_grade="A",
        total_checks=25,
        passed_checks=23,
        warning_count=2,
        error_count=0,
        critical_count=0,
        execution_time=0.48,
        ready_for_training=True,
        recommendations=["Impute missing values"],
    )

    d = summary.to_dict()
    assert d["passed"] is True
    assert d["health_score"] == 94
    assert d["health_grade"] == "A"
    assert d["ready_for_training"] is True

    json_str = summary.to_json()
    parsed = json.loads(json_str)
    assert parsed["health_score"] == 94


def test_summary_text_formatting():
    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    pipeline = ValidationPipeline()
    summary = pipeline.validate(df, target="target")

    text = summary.summary_text()
    assert "🪁 KiteML Validation Report" in text
    assert "DatasetValidator" in text
    assert "Status" in text
    assert "READY FOR TRAINING" in text
