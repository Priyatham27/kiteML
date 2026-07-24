"""
test_blueprint.py — Unit tests for FeatureEngineeringBlueprint (Story 4.2).
"""

import pytest

from kiteml.feature_engineering import (
    EngineeredFeaturePlan,
    FeatureEngineeringBlueprint,
    FETransformType,
)


def test_engineered_feature_plan_serialization():
    plan = EngineeredFeaturePlan(
        generated_name="revenue",
        source_columns=["price", "quantity"],
        transform_type=FETransformType.INTERACTION_PRODUCT,
        provider_name="InteractionFEProvider",
        estimated_importance=0.95,
        reasoning=["Multiplicative interaction."],
    )

    d = plan.to_dict()
    assert d["generated_name"] == "revenue"
    assert d["star_rating"] == "★★★★★"
    assert d["transform_type"] == "interaction_product"


def test_feature_engineering_blueprint_summary():
    plan1 = EngineeredFeaturePlan(
        generated_name="date_year",
        source_columns=["date"],
        transform_type=FETransformType.DATETIME_YEAR,
        provider_name="DatetimeFEProvider",
        estimated_importance=0.85,
    )

    bp = FeatureEngineeringBlueprint(
        feature_plans={"date_year": plan1},
        target_name="sales",
        problem_type="regression",
    )

    summary = bp.summary_text()
    assert "⚙️ KiteML Feature Engineering Blueprint" in summary
    assert "date_year" in summary
    assert "★★★★☆" in summary

    d = bp.to_dict()
    assert d["target_name"] == "sales"
    assert d["feature_count"] == 1
