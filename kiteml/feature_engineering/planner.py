"""
planner.py — FeatureEngineeringEngine planner orchestrator for KiteML.
"""

from typing import Any, Sequence

import pandas as pd

from kiteml.feature_engineering.blueprint import EngineeredFeaturePlan, FeatureEngineeringBlueprint
from kiteml.feature_engineering.importance_predictor import FeatureImportancePredictor
from kiteml.feature_engineering.providers import (
    BaseFEProvider,
    CategoricalFEProvider,
    DatetimeFEProvider,
    InteractionFEProvider,
    NumericFEProvider,
    TextFEProvider,
)
from kiteml.feature_engineering.rules import FERuleEngine, default_fe_rule_engine
from kiteml.intelligence.data_profiler import build_data_profile


class FeatureEngineeringEngine:
    """
    Intelligent Feature Engineering Engine planner.

    Analyzes dataset feature patterns and automatically identifies opportunities
    to engineer candidate features (datetime extractions, numeric transforms,
    interactions, text statistics), returning a FeatureEngineeringBlueprint.
    """

    def __init__(
        self,
        providers: Sequence[BaseFEProvider] | None = None,
        rules: FERuleEngine | None = None,
        predictor: FeatureImportancePredictor | None = None,
    ) -> None:
        self.rules = rules or default_fe_rule_engine
        self.predictor = predictor or FeatureImportancePredictor()
        self.providers: list[BaseFEProvider] = list(providers) if providers is not None else self._default_providers()

    def _default_providers(self) -> list[BaseFEProvider]:
        """Return standard suite of feature engineering providers."""
        return [
            DatetimeFEProvider(),
            NumericFEProvider(),
            InteractionFEProvider(),
            CategoricalFEProvider(),
            TextFEProvider(),
        ]

    def register_provider(self, provider: BaseFEProvider) -> None:
        """Register a custom feature engineering provider."""
        self.providers.append(provider)

    def plan(
        self,
        df: pd.DataFrame,
        preprocessing_blueprint: Any | None = None,
        target: str | None = None,
        problem_type: str | None = None,
    ) -> FeatureEngineeringBlueprint:
        """
        Generate a FeatureEngineeringBlueprint for the input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.
        preprocessing_blueprint : PreprocessingBlueprint, optional
            Output blueprint from Story 4.1.
        target : str, optional
            Target feature column name.
        problem_type : str, optional
            Task type ('classification' or 'regression').

        Returns
        -------
        FeatureEngineeringBlueprint
            Structured feature engineering plan.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            return FeatureEngineeringBlueprint(target_name=target, problem_type=problem_type)

        target_col = target or (str(df.columns[-1]) if len(df.columns) > 0 else "")
        prob_type = problem_type or "classification"

        data_prof = None
        if target_col in df.columns:
            try:
                data_prof = build_data_profile(df, target=target_col, problem_type=prob_type)
            except Exception:
                data_prof = None

        plans: dict[str, EngineeredFeaturePlan] = {}

        for provider in self.providers:
            prov_plans = provider.plan_features(
                df=df,
                data_profile=data_prof,
                rules=self.rules,
                predictor=self.predictor,
                target=target,
            )
            for plan in prov_plans:
                plans[plan.generated_name] = plan

        return FeatureEngineeringBlueprint(
            feature_plans=plans,
            target_name=target,
            problem_type=problem_type,
        )


# Shortcut alias
fe_engine = FeatureEngineeringEngine()
