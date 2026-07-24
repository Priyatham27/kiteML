"""
providers.py — Strategy providers for missing values, encoding, scaling, datetime, and text.
"""

from abc import ABC, abstractmethod
from typing import Any

from kiteml.preprocessing.blueprint import FeaturePlan
from kiteml.preprocessing.rules import RuleEngine
from kiteml.preprocessing.strategy import (
    DatetimeStrategy,
    EncodingStrategy,
    MissingStrategy,
    ScalingStrategy,
    TextStrategy,
)


class BaseStrategyProvider(ABC):
    """Abstract base class for preprocessing strategy providers."""

    name: str = "BaseProvider"

    @abstractmethod
    def apply(
        self,
        feature_name: str,
        profile: dict[str, Any],
        plan: FeaturePlan,
        rules: RuleEngine,
    ) -> None:
        """Apply strategy rules and update FeaturePlan in place."""
        pass


class MissingValueProvider(BaseStrategyProvider):
    """Provider planning missing value imputation strategies."""

    name = "MissingValueProvider"

    def apply(
        self,
        feature_name: str,
        profile: dict[str, Any],
        plan: FeaturePlan,
        rules: RuleEngine,
    ) -> None:
        missing_count = profile.get("missing_count", 0)
        missing_ratio = profile.get("missing_ratio", 0.0)

        if missing_count == 0:
            plan.missing_strategy = MissingStrategy.NONE
            return

        if missing_ratio >= rules.high_missing_drop_threshold:
            plan.ignore = True
            plan.missing_strategy = MissingStrategy.DROP
            plan.reasoning.append(
                f"Ignored feature: missing ratio ({missing_ratio:.1%}) exceeds threshold ({rules.high_missing_drop_threshold:.0%})."
            )
            return

        is_numeric = profile.get("is_numeric", False)
        if is_numeric:
            plan.missing_strategy = MissingStrategy.MEDIAN
            plan.reasoning.append(f"Impute missing values using Median (missing ratio: {missing_ratio:.1%}).")
        else:
            plan.missing_strategy = MissingStrategy.MODE
            plan.reasoning.append(
                f"Impute missing categorical values using Most Frequent Mode (missing ratio: {missing_ratio:.1%})."
            )


class EncodingProvider(BaseStrategyProvider):
    """Provider planning categorical encoding strategies."""

    name = "EncodingProvider"

    def apply(
        self,
        feature_name: str,
        profile: dict[str, Any],
        plan: FeaturePlan,
        rules: RuleEngine,
    ) -> None:
        if plan.ignore:
            return

        is_categorical = (
            profile.get("is_categorical", False)
            or str(profile.get("datatype", "")).startswith("string")
            or profile.get("datatype") in ("object", "category", "string")
        )
        is_text = profile.get("is_text", False)
        is_datetime = profile.get("is_datetime", False)

        if not is_categorical or is_text or is_datetime:
            plan.encoding_strategy = EncodingStrategy.NONE
            return

        nunique = profile.get("nunique", 0)

        if nunique == 1:
            plan.ignore = True
            plan.encoding_strategy = EncodingStrategy.NONE
            plan.reasoning.append("Ignored feature: 0 variance (constant feature).")
            return

        if nunique <= rules.low_cardinality_threshold:
            plan.encoding_strategy = EncodingStrategy.ONE_HOT
            plan.reasoning.append(
                f"One-Hot Encode categorical feature ({nunique} unique categories <= {rules.low_cardinality_threshold})."
            )
        else:
            plan.encoding_strategy = EncodingStrategy.TARGET
            plan.reasoning.append(
                f"Target Encode high cardinality categorical feature ({nunique} unique categories > {rules.low_cardinality_threshold})."
            )


class ScalingProvider(BaseStrategyProvider):
    """Provider planning numerical feature scaling strategies."""

    name = "ScalingProvider"

    def apply(
        self,
        feature_name: str,
        profile: dict[str, Any],
        plan: FeaturePlan,
        rules: RuleEngine,
    ) -> None:
        if plan.ignore:
            return

        is_numeric = profile.get("is_numeric", False)
        if not is_numeric:
            plan.scaling_strategy = ScalingStrategy.NONE
            return

        skewness = profile.get("skewness", 0.0)

        if abs(skewness) > rules.high_skewness_threshold:
            plan.scaling_strategy = ScalingStrategy.ROBUST
            plan.reasoning.append(f"RobustScaler applied due to high distribution skewness ({skewness:.2f}).")
        else:
            plan.scaling_strategy = ScalingStrategy.STANDARD
            plan.reasoning.append("StandardScaler applied to numeric feature.")


class DatetimeProvider(BaseStrategyProvider):
    """Provider planning datetime feature extraction."""

    name = "DatetimeProvider"

    def apply(
        self,
        feature_name: str,
        profile: dict[str, Any],
        plan: FeaturePlan,
        rules: RuleEngine,
    ) -> None:
        if plan.ignore:
            return

        is_datetime = profile.get("is_datetime", False)
        if is_datetime:
            plan.datetime_strategy = DatetimeStrategy.EXTRACT_COMPONENTS
            plan.encoding_strategy = EncodingStrategy.NONE
            plan.scaling_strategy = ScalingStrategy.NONE
            plan.reasoning.append("Extract datetime components (year, month, day, weekday, quarter).")


class TextProvider(BaseStrategyProvider):
    """Provider planning text feature vectorization."""

    name = "TextProvider"

    def apply(
        self,
        feature_name: str,
        profile: dict[str, Any],
        plan: FeaturePlan,
        rules: RuleEngine,
    ) -> None:
        if plan.ignore:
            return

        is_text = profile.get("is_text", False)
        if is_text:
            plan.text_strategy = TextStrategy.TFIDF
            plan.encoding_strategy = EncodingStrategy.NONE
            plan.scaling_strategy = ScalingStrategy.NONE
            plan.reasoning.append("Free text feature vectorization using TF-IDF.")
