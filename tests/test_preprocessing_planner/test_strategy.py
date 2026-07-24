"""
test_strategy.py — Unit tests for strategy enums and strategy providers (Story 4.1).
"""

import pytest

from kiteml.preprocessing import (
    EncodingProvider,
    EncodingStrategy,
    FeaturePlan,
    MissingStrategy,
    MissingValueProvider,
    RuleEngine,
    ScalingProvider,
    ScalingStrategy,
)


def test_strategy_provider_missing():
    provider = MissingValueProvider()
    rules = RuleEngine()
    plan = FeaturePlan(feature_name="test_col", datatype="float64")

    # High missing > 80%
    prof = {"missing_count": 90, "missing_ratio": 0.90, "is_numeric": True}
    provider.apply("test_col", prof, plan, rules)
    assert plan.ignore is True
    assert plan.missing_strategy == MissingStrategy.DROP


def test_strategy_provider_encoding():
    provider = EncodingProvider()
    rules = RuleEngine(low_cardinality_threshold=5)

    # Low cardinality <= 5
    plan1 = FeaturePlan(feature_name="cat1", datatype="object")
    prof1 = {"is_categorical": True, "nunique": 3}
    provider.apply("cat1", prof1, plan1, rules)
    assert plan1.encoding_strategy == EncodingStrategy.ONE_HOT

    # High cardinality > 5
    plan2 = FeaturePlan(feature_name="cat2", datatype="object")
    prof2 = {"is_categorical": True, "nunique": 20}
    provider.apply("cat2", prof2, plan2, rules)
    assert plan2.encoding_strategy == EncodingStrategy.TARGET


def test_strategy_provider_scaling():
    provider = ScalingProvider()
    rules = RuleEngine(high_skewness_threshold=1.5)

    # High skew -> RobustScaler
    plan = FeaturePlan(feature_name="num", datatype="float64")
    prof = {"is_numeric": True, "skewness": 3.2}
    provider.apply("num", prof, plan, rules)
    assert plan.scaling_strategy == ScalingStrategy.ROBUST
