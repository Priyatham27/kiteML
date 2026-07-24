"""
planner.py — PreprocessingEngine planner orchestrator for KiteML.
"""

from typing import Any, Sequence

import pandas as pd

from kiteml.intelligence.data_profiler import build_data_profile
from kiteml.preprocessing.blueprint import FeaturePlan, PreprocessingBlueprint
from kiteml.preprocessing.providers import (
    BaseStrategyProvider,
    DatetimeProvider,
    EncodingProvider,
    MissingValueProvider,
    ScalingProvider,
    TextProvider,
)
from kiteml.preprocessing.rules import RuleEngine, default_rule_engine


class PreprocessingEngine:
    """
    Intelligent Preprocessing Engine planner.

    Inspects dataset feature profiles and automatically selects missing value,
    encoding, scaling, datetime, and text strategies, returning a PreprocessingBlueprint.
    """

    def __init__(
        self,
        providers: Sequence[BaseStrategyProvider] | None = None,
        rules: RuleEngine | None = None,
    ) -> None:
        self.rules = rules or default_rule_engine
        self.providers: list[BaseStrategyProvider] = (
            list(providers) if providers is not None else self._default_providers()
        )

    def _default_providers(self) -> list[BaseStrategyProvider]:
        """Return standard suite of preprocessing strategy providers."""
        return [
            MissingValueProvider(),
            EncodingProvider(),
            ScalingProvider(),
            DatetimeProvider(),
            TextProvider(),
        ]

    def register_provider(self, provider: BaseStrategyProvider) -> None:
        """Register a custom strategy provider."""
        self.providers.append(provider)

    def plan(
        self,
        df: pd.DataFrame,
        target: str | None = None,
        problem_type: str | None = None,
    ) -> PreprocessingBlueprint:
        """
        Generate a PreprocessingBlueprint for the input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.
        target : str, optional
            Target feature column name.
        problem_type : str, optional
            Task type ('classification' or 'regression').

        Returns
        -------
        PreprocessingBlueprint
            Structured preprocessing execution plan.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            return PreprocessingBlueprint(target_name=target, problem_type=problem_type)

        target_col = target or (str(df.columns[-1]) if len(df.columns) > 0 else "")
        prob_type = problem_type or "classification"

        data_prof = None
        if target_col in df.columns:
            try:
                data_prof = build_data_profile(df, target=target_col, problem_type=prob_type)
            except Exception:
                data_prof = None

        feature_plans: dict[str, FeaturePlan] = {}

        for col in df.columns:
            if target and col == target:
                continue

            is_text = False
            is_datetime = pd.api.types.is_datetime64_any_dtype(df[col].dtype)

            if data_prof:
                if hasattr(data_prof, "text") and hasattr(data_prof.text, "text_columns"):
                    is_text = col in data_prof.text.text_columns
                if hasattr(data_prof, "datetime") and hasattr(data_prof.datetime, "datetime_columns"):
                    is_datetime = is_datetime or (col in data_prof.datetime.datetime_columns)

            raw_col_skew = (
                df[col].skew() if pd.api.types.is_numeric_dtype(df[col].dtype) and df[col].nunique() > 1 else 0.0
            )
            feat_prof: dict[str, Any] = {
                "datatype": str(df[col].dtype),
                "nunique": df[col].nunique(dropna=True),
                "missing_count": int(df[col].isna().sum()),
                "missing_ratio": float(df[col].isna().mean()),
                "is_numeric": pd.api.types.is_numeric_dtype(df[col].dtype),
                "is_categorical": (
                    pd.api.types.is_object_dtype(df[col].dtype)
                    or pd.api.types.is_string_dtype(df[col].dtype)
                    or isinstance(df[col].dtype, pd.CategoricalDtype)
                ),
                "is_datetime": is_datetime,
                "is_text": is_text,
                "skewness": float(raw_col_skew) if isinstance(raw_col_skew, (int, float)) else 0.0,
            }

            # Create base FeaturePlan
            plan = FeaturePlan(
                feature_name=col,
                datatype=feat_prof["datatype"],
            )

            # Apply all strategy providers
            for provider in self.providers:
                provider.apply(col, feat_prof, plan, self.rules)

            feature_plans[col] = plan

        return PreprocessingBlueprint(
            feature_plans=feature_plans,
            target_name=target,
            problem_type=problem_type,
            global_settings={
                "low_cardinality_threshold": self.rules.low_cardinality_threshold,
                "high_missing_drop_threshold": self.rules.high_missing_drop_threshold,
                "high_skewness_threshold": self.rules.high_skewness_threshold,
            },
        )


# Shortcut alias
engine = PreprocessingEngine()
