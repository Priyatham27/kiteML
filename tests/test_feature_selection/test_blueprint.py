"""
test_blueprint.py — Unit tests for FeatureSelectionBlueprint (Story 4.3).
"""

import pytest

from kiteml.feature_selection import (
    FeatureScore,
    FeatureSelectionBlueprint,
    SelectionDecision,
)


def test_feature_score_serialization():
    score = FeatureScore(
        feature_name="age",
        score=92.5,
        confidence=0.95,
        decision=SelectionDecision.KEEP,
        reasoning=["Passed all selector checks."],
    )

    d = score.to_dict()
    assert d["feature_name"] == "age"
    assert d["score"] == 92.5
    assert d["decision"] == "keep"


def test_feature_selection_blueprint_summary():
    score1 = FeatureScore(feature_name="age", score=90.0, decision=SelectionDecision.KEEP)
    score2 = FeatureScore(
        feature_name="constant_col",
        score=0.0,
        decision=SelectionDecision.REMOVE,
        reasoning=["Constant feature (0 variance)."],
    )

    bp = FeatureSelectionBlueprint(
        selected_features=["age"],
        removed_features=["constant_col"],
        feature_scores={"age": score1, "constant_col": score2},
        target_name="price",
        problem_type="regression",
    )

    summary = bp.summary_text()
    assert "🎯 KiteML Feature Selection Blueprint" in summary
    assert "age" in summary
    assert "constant_col" in summary

    d = bp.to_dict()
    assert d["target_name"] == "price"
    assert d["total_features"] == 2
