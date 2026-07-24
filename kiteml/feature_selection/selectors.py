"""
selectors.py — Feature selectors for rule-based, variance, missing value, correlation, and importance evaluation.
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from kiteml.feature_selection.rules import FSRuleEngine
from kiteml.feature_selection.strategy import SelectionDecision


class BaseSelector(ABC):
    """Abstract base class for feature selection evaluators."""

    name: str = "BaseSelector"

    @abstractmethod
    def evaluate(
        self,
        col: str,
        df: pd.DataFrame,
        data_profile: Any,
        rules: FSRuleEngine,
        target: str | None = None,
    ) -> tuple[SelectionDecision, float, str]:
        """
        Evaluate feature and return (decision, score_0_to_100, reasoning_text).
        """
        pass


class RuleSelector(BaseSelector):
    """Selector identifying constant features, empty columns, and identifier features."""

    name = "RuleSelector"

    def evaluate(
        self,
        col: str,
        df: pd.DataFrame,
        data_profile: Any,
        rules: FSRuleEngine,
        target: str | None = None,
    ) -> tuple[SelectionDecision, float, str]:
        if df[col].empty:
            return (SelectionDecision.REMOVE, 0.0, "Empty feature column.")

        nunique = df[col].nunique(dropna=True)
        if nunique <= 1:
            return (SelectionDecision.REMOVE, 0.0, "Constant feature (0 variance).")

        col_lower = col.lower()
        if col_lower in ("id", "customer_id", "customerid", "user_id", "userid", "guid", "uuid") or col_lower.endswith(
            "_id"
        ):
            if nunique >= len(df) * 0.90:
                return (SelectionDecision.REMOVE, 5.0, f"Identifier column '{col}' with high unique ratio.")

        return (SelectionDecision.KEEP, 100.0, "Passed structural rule checks.")


class VarianceSelector(BaseSelector):
    """Selector checking for near-zero numerical variance."""

    name = "VarianceSelector"

    def evaluate(
        self,
        col: str,
        df: pd.DataFrame,
        data_profile: Any,
        rules: FSRuleEngine,
        target: str | None = None,
    ) -> tuple[SelectionDecision, float, str]:
        if not pd.api.types.is_numeric_dtype(df[col].dtype):
            return (SelectionDecision.KEEP, 100.0, "Non-numeric feature exempt from variance check.")

        var_val = float(df[col].var()) if len(df) > 1 else 0.0
        if var_val < rules.min_variance:
            return (SelectionDecision.REMOVE, 10.0, f"Near-zero variance ({var_val:.2e} < {rules.min_variance:.2e}).")

        return (SelectionDecision.KEEP, 90.0, f"Adequate numerical variance ({var_val:.4f}).")


class MissingValueSelector(BaseSelector):
    """Selector checking for excessive missing value ratios."""

    name = "MissingValueSelector"

    def evaluate(
        self,
        col: str,
        df: pd.DataFrame,
        data_profile: Any,
        rules: FSRuleEngine,
        target: str | None = None,
    ) -> tuple[SelectionDecision, float, str]:
        missing_ratio = float(df[col].isna().mean())
        if missing_ratio >= rules.max_missing_ratio:
            return (
                SelectionDecision.REMOVE,
                15.0,
                f"Missing ratio ({missing_ratio:.1%}) exceeds threshold ({rules.max_missing_ratio:.0%}).",
            )

        score = max(20.0, 100.0 - (missing_ratio * 100.0))
        return (SelectionDecision.KEEP, score, f"Acceptable missing ratio ({missing_ratio:.1%}).")


class CorrelationSelector(BaseSelector):
    """Selector checking for high pairwise collinearity."""

    name = "CorrelationSelector"

    def evaluate(
        self,
        col: str,
        df: pd.DataFrame,
        data_profile: Any,
        rules: FSRuleEngine,
        target: str | None = None,
    ) -> tuple[SelectionDecision, float, str]:
        if not pd.api.types.is_numeric_dtype(df[col].dtype):
            return (SelectionDecision.KEEP, 100.0, "Non-numeric feature exempt from correlation check.")

        # Check correlations if data_profile contains correlation matrix
        if data_profile and hasattr(data_profile, "correlations") and hasattr(data_profile.correlations, "matrix"):
            matrix = data_profile.correlations.matrix
            if isinstance(matrix, pd.DataFrame) and col in matrix.columns:
                for other_col in matrix.columns:
                    if other_col == col or (target and other_col == target):
                        continue
                    val = matrix.loc[col, other_col]
                    if isinstance(val, (int, float)):
                        corr_val = abs(float(val))
                        if corr_val >= rules.max_correlation:
                            return (
                                SelectionDecision.REMOVE,
                                20.0,
                                f"High collinearity ({corr_val:.2f}) with feature '{other_col}'.",
                            )

        return (SelectionDecision.KEEP, 90.0, "No extreme collinearity detected.")


class ImportanceEstimatorSelector(BaseSelector):
    """Selector estimating feature importance and target relevance."""

    name = "ImportanceEstimator"

    def evaluate(
        self,
        col: str,
        df: pd.DataFrame,
        data_profile: Any,
        rules: FSRuleEngine,
        target: str | None = None,
    ) -> tuple[SelectionDecision, float, str]:
        if (
            target
            and target in df.columns
            and pd.api.types.is_numeric_dtype(df[col].dtype)
            and pd.api.types.is_numeric_dtype(df[target].dtype)
        ):
            try:
                raw_corr = df[col].corr(df[target])
                if isinstance(raw_corr, (int, float)) and not pd.isna(raw_corr):
                    corr = abs(float(raw_corr))
                    score = min(100.0, max(40.0, corr * 100.0 + 30.0))
                    return (SelectionDecision.KEEP, score, f"Target correlation magnitude ({corr:.2f}).")
            except Exception:
                pass

        return (SelectionDecision.KEEP, 75.0, "Default baseline feature importance estimation.")
