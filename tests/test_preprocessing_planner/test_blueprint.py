"""
test_blueprint.py — Unit tests for PreprocessingBlueprint and FeaturePlan (Story 4.1).
"""

import pytest

from kiteml.preprocessing import (
    EncodingStrategy,
    FeaturePlan,
    MissingStrategy,
    PreprocessingBlueprint,
    ScalingStrategy,
)


def test_feature_plan_serialization():
    plan = FeaturePlan(
        feature_name="age",
        datatype="int64",
        missing_strategy=MissingStrategy.MEDIAN,
        scaling_strategy=ScalingStrategy.STANDARD,
        reasoning=["Impute missing values using Median."],
    )
    d = plan.to_dict()

    assert d["feature_name"] == "age"
    assert d["missing_strategy"] == "median"
    assert d["scaling_strategy"] == "standard"
    assert "Impute missing values" in d["reasoning"][0]


def test_blueprint_summary_and_serialization():
    plan1 = FeaturePlan(
        feature_name="income",
        datatype="float64",
        scaling_strategy=ScalingStrategy.STANDARD,
    )
    plan2 = FeaturePlan(
        feature_name="constant_col",
        datatype="object",
        ignore=True,
        reasoning=["Ignored feature: 0 variance."],
    )

    bp = PreprocessingBlueprint(
        feature_plans={"income": plan1, "constant_col": plan2},
        target_name="price",
        problem_type="regression",
    )

    assert bp.feature_count == 2
    assert bp.active_features == ["income"]
    assert bp.ignored_features == ["constant_col"]

    summary = bp.summary_text()
    assert "📋 KiteML Preprocessing Blueprint" in summary
    assert "income" in summary
    assert "constant_col" in summary

    d = bp.to_dict()
    assert d["target_name"] == "price"
    assert d["feature_count"] == 2
