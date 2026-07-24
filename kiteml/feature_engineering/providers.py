"""
providers.py — Specialized feature engineering strategy providers.
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from kiteml.feature_engineering.blueprint import EngineeredFeaturePlan
from kiteml.feature_engineering.importance_predictor import FeatureImportancePredictor
from kiteml.feature_engineering.rules import FERuleEngine
from kiteml.feature_engineering.strategy import FETransformType


class BaseFEProvider(ABC):
    """Abstract base class for feature engineering providers."""

    name: str = "BaseFEProvider"

    @abstractmethod
    def plan_features(
        self,
        df: pd.DataFrame,
        data_profile: Any,
        rules: FERuleEngine,
        predictor: FeatureImportancePredictor,
        target: str | None = None,
    ) -> list[EngineeredFeaturePlan]:
        """Generate candidate engineered feature plans."""
        pass


class DatetimeFEProvider(BaseFEProvider):
    """Provider proposing datetime component features."""

    name = "DatetimeFEProvider"

    def plan_features(
        self,
        df: pd.DataFrame,
        data_profile: Any,
        rules: FERuleEngine,
        predictor: FeatureImportancePredictor,
        target: str | None = None,
    ) -> list[EngineeredFeaturePlan]:
        if not rules.enable_datetime_extractions:
            return []

        plans: list[EngineeredFeaturePlan] = []

        for col in df.columns:
            if target and col == target:
                continue

            is_dt = pd.api.types.is_datetime64_any_dtype(df[col].dtype)
            if (
                not is_dt
                and data_profile
                and hasattr(data_profile, "datetime")
                and hasattr(data_profile.datetime, "datetime_columns")
            ):
                is_dt = col in data_profile.datetime.datetime_columns

            if not is_dt:
                continue

            transforms = [
                (f"{col}_year", FETransformType.DATETIME_YEAR),
                (f"{col}_month", FETransformType.DATETIME_MONTH),
                (f"{col}_day", FETransformType.DATETIME_DAY),
                (f"{col}_weekday", FETransformType.DATETIME_WEEKDAY),
                (f"{col}_quarter", FETransformType.DATETIME_QUARTER),
                (f"{col}_is_weekend", FETransformType.DATETIME_IS_WEEKEND),
            ]

            for gen_name, tt in transforms:
                imp, conf, reason = predictor.predict([col], tt, gen_name)
                plans.append(
                    EngineeredFeaturePlan(
                        generated_name=gen_name,
                        source_columns=[col],
                        transform_type=tt,
                        provider_name=self.name,
                        confidence=conf,
                        estimated_importance=imp,
                        reasoning=[reason],
                    )
                )

        return plans


class NumericFEProvider(BaseFEProvider):
    """Provider proposing numeric transformations (Log, Sqrt, Square)."""

    name = "NumericFEProvider"

    def plan_features(
        self,
        df: pd.DataFrame,
        data_profile: Any,
        rules: FERuleEngine,
        predictor: FeatureImportancePredictor,
        target: str | None = None,
    ) -> list[EngineeredFeaturePlan]:
        if not rules.enable_numeric_transforms:
            return []

        plans: list[EngineeredFeaturePlan] = []

        for col in df.columns:
            if target and col == target:
                continue

            if not pd.api.types.is_numeric_dtype(df[col].dtype):
                continue

            if df[col].nunique() <= 1:
                continue

            raw_skew = df[col].skew() if df[col].nunique() > 1 else 0.0
            skew = float(raw_skew) if isinstance(raw_skew, (int, float)) else 0.0
            raw_min = df[col].min() if not df[col].empty else 0.0
            min_val = float(raw_min) if isinstance(raw_min, (int, float)) else 0.0

            # Right-skewed positive data -> Log transform candidate
            if skew >= rules.skewness_threshold and min_val >= 0.0:
                gen_name = f"log_{col}"
                tt = FETransformType.LOG_TRANSFORM
                imp, conf, reason = predictor.predict([col], tt, gen_name)
                plans.append(
                    EngineeredFeaturePlan(
                        generated_name=gen_name,
                        source_columns=[col],
                        transform_type=tt,
                        provider_name=self.name,
                        confidence=conf,
                        estimated_importance=imp,
                        reasoning=[reason],
                    )
                )

        return plans


class InteractionFEProvider(BaseFEProvider):
    """Provider proposing pairwise interactions for numeric features."""

    name = "InteractionFEProvider"

    def plan_features(
        self,
        df: pd.DataFrame,
        data_profile: Any,
        rules: FERuleEngine,
        predictor: FeatureImportancePredictor,
        target: str | None = None,
    ) -> list[EngineeredFeaturePlan]:
        if not rules.enable_interactions:
            return []

        num_cols = [
            col
            for col in df.columns
            if (target is None or col != target)
            and pd.api.types.is_numeric_dtype(df[col].dtype)
            and df[col].nunique() > 1
        ]

        if len(num_cols) < 2:
            return []

        plans: list[EngineeredFeaturePlan] = []
        pairs_evaluated = 0

        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                if pairs_evaluated >= rules.max_interaction_pairs:
                    break

                col1, col2 = num_cols[i], num_cols[j]
                pairs_evaluated += 1

                gen_name = f"{col1}_x_{col2}"
                tt = FETransformType.INTERACTION_PRODUCT
                imp, conf, reason = predictor.predict([col1, col2], tt, gen_name)
                plans.append(
                    EngineeredFeaturePlan(
                        generated_name=gen_name,
                        source_columns=[col1, col2],
                        transform_type=tt,
                        provider_name=self.name,
                        confidence=conf,
                        estimated_importance=imp,
                        reasoning=[reason],
                    )
                )

        return plans


class CategoricalFEProvider(BaseFEProvider):
    """Provider proposing categorical frequency encodings."""

    name = "CategoricalFEProvider"

    def plan_features(
        self,
        df: pd.DataFrame,
        data_profile: Any,
        rules: FERuleEngine,
        predictor: FeatureImportancePredictor,
        target: str | None = None,
    ) -> list[EngineeredFeaturePlan]:
        plans: list[EngineeredFeaturePlan] = []

        for col in df.columns:
            if target and col == target:
                continue

            is_cat = pd.api.types.is_object_dtype(df[col].dtype) or isinstance(df[col].dtype, pd.CategoricalDtype)
            if is_cat and df[col].nunique() > 10:
                gen_name = f"{col}_freq"
                tt = FETransformType.CATEGORY_FREQUENCY
                imp, conf, reason = predictor.predict([col], tt, gen_name)
                plans.append(
                    EngineeredFeaturePlan(
                        generated_name=gen_name,
                        source_columns=[col],
                        transform_type=tt,
                        provider_name=self.name,
                        confidence=conf,
                        estimated_importance=imp,
                        reasoning=[reason],
                    )
                )

        return plans


class TextFEProvider(BaseFEProvider):
    """Provider proposing text length statistics (word count, char count)."""

    name = "TextFEProvider"

    def plan_features(
        self,
        df: pd.DataFrame,
        data_profile: Any,
        rules: FERuleEngine,
        predictor: FeatureImportancePredictor,
        target: str | None = None,
    ) -> list[EngineeredFeaturePlan]:
        if not rules.enable_text_derived:
            return []

        plans: list[EngineeredFeaturePlan] = []

        for col in df.columns:
            if target and col == target:
                continue

            is_text = False
            if data_profile and hasattr(data_profile, "text") and hasattr(data_profile.text, "text_columns"):
                is_text = col in data_profile.text.text_columns

            if is_text:
                transforms = [
                    (f"{col}_word_count", FETransformType.TEXT_WORD_COUNT),
                    (f"{col}_char_count", FETransformType.TEXT_CHAR_COUNT),
                ]

                for gen_name, tt in transforms:
                    imp, conf, reason = predictor.predict([col], tt, gen_name)
                    plans.append(
                        EngineeredFeaturePlan(
                            generated_name=gen_name,
                            source_columns=[col],
                            transform_type=tt,
                            provider_name=self.name,
                            confidence=conf,
                            estimated_importance=imp,
                            reasoning=[reason],
                        )
                    )

        return plans
