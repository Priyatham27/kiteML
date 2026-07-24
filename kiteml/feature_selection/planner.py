"""
planner.py — FeatureSelectionEngine planner orchestrator for KiteML.
"""

from typing import Any, Sequence

import pandas as pd

from kiteml.feature_selection.blueprint import FeatureScore, FeatureSelectionBlueprint
from kiteml.feature_selection.rules import FSRuleEngine, default_fs_rule_engine
from kiteml.feature_selection.selectors import (
    BaseSelector,
    CorrelationSelector,
    ImportanceEstimatorSelector,
    MissingValueSelector,
    RuleSelector,
    VarianceSelector,
)
from kiteml.feature_selection.voting import FeatureSelectionVotingSystem
from kiteml.intelligence.data_profiler import build_data_profile


class FeatureSelectionEngine:
    """
    Intelligent Feature Selection Engine planner.

    Evaluates dataset features using an ensemble voting system (Rule, Variance,
    MissingValue, Correlation, and Importance selectors) and produces a FeatureSelectionBlueprint.
    """

    def __init__(
        self,
        selectors: Sequence[BaseSelector] | None = None,
        rules: FSRuleEngine | None = None,
        voting_system: FeatureSelectionVotingSystem | None = None,
    ) -> None:
        self.rules = rules or default_fs_rule_engine
        self.voting_system = voting_system or FeatureSelectionVotingSystem()
        self.selectors: list[BaseSelector] = list(selectors) if selectors is not None else self._default_selectors()

    def _default_selectors(self) -> list[BaseSelector]:
        """Return standard suite of feature selection evaluators."""
        return [
            RuleSelector(),
            VarianceSelector(),
            MissingValueSelector(),
            CorrelationSelector(),
            ImportanceEstimatorSelector(),
        ]

    def register_selector(self, selector: BaseSelector) -> None:
        """Register a custom feature selector."""
        self.selectors.append(selector)

    def plan(
        self,
        df: pd.DataFrame,
        preprocessing_blueprint: Any | None = None,
        feature_engineering_blueprint: Any | None = None,
        target: str | None = None,
        problem_type: str | None = None,
        keep_features: Sequence[str] | None = None,
    ) -> FeatureSelectionBlueprint:
        """
        Generate a FeatureSelectionBlueprint for the input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.
        preprocessing_blueprint : PreprocessingBlueprint, optional
            Output blueprint from Story 4.1.
        feature_engineering_blueprint : FeatureEngineeringBlueprint, optional
            Output blueprint from Story 4.2.
        target : str, optional
            Target feature column name.
        problem_type : str, optional
            Task type ('classification' or 'regression').
        keep_features : Sequence[str], optional
            Protected feature names requested to be retained.

        Returns
        -------
        FeatureSelectionBlueprint
            Structured feature selection plan.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            return FeatureSelectionBlueprint(target_name=target, problem_type=problem_type)

        target_col = target or (str(df.columns[-1]) if len(df.columns) > 0 else "")
        prob_type = problem_type or "classification"
        protected_list = list(keep_features) if keep_features else []

        data_prof = None
        if target_col in df.columns:
            try:
                data_prof = build_data_profile(df, target=target_col, problem_type=prob_type)
            except Exception:
                data_prof = None

        feature_scores: dict[str, FeatureScore] = {}
        selected_features: list[str] = []
        removed_features: list[str] = []

        for col in df.columns:
            if target and col == target:
                continue

            score_obj = self.voting_system.evaluate_feature(
                col=col,
                df=df,
                selectors=self.selectors,
                rules=self.rules,
                data_profile=data_prof,
                target=target,
                protected_features=protected_list,
            )

            feature_scores[col] = score_obj

            dec_str = score_obj.decision.value if hasattr(score_obj.decision, "value") else str(score_obj.decision)
            if dec_str == "remove":
                removed_features.append(col)
            else:
                selected_features.append(col)

        return FeatureSelectionBlueprint(
            selected_features=selected_features,
            removed_features=removed_features,
            protected_features=protected_list,
            feature_scores=feature_scores,
            target_name=target,
            problem_type=problem_type,
        )


# Shortcut alias
fs_engine = FeatureSelectionEngine()
