"""
target_validator.py — Composite Target Validator for KiteML.
"""

from typing import Any

import pandas as pd

from kiteml.validation.rules.target_rules import (
    ClassImbalanceRule,
    ConstantTargetRule,
    ExcessiveClassesRule,
    IdentifierTargetRule,
    MinClassesClassificationRule,
    NumericRegressionTargetRule,
    TargetContainsValuesRule,
    TargetExistsRule,
    TargetMissingPercentageRule,
    TargetSpecifiedRule,
)
from kiteml.validation.validation_result import ValidationResult
from kiteml.validation.validator import BaseValidator


class TargetValidator(BaseValidator):
    """
    Validates target column existence, missingness, problem type suitability,
    class distribution / balance for classification, numeric variance for regression,
    and identifier patterns. Generates Target Intelligence metadata.
    """

    description: str = "Validates target column suitability for machine learning."

    @property
    def name(self) -> str:
        return "TargetValidator"

    def __init__(self, rules: list[Any] | None = None) -> None:
        if rules is None:
            rules = [
                TargetSpecifiedRule(),
                TargetExistsRule(),
                TargetContainsValuesRule(),
                TargetMissingPercentageRule(),
                MinClassesClassificationRule(),
                ClassImbalanceRule(),
                ConstantTargetRule(),
                NumericRegressionTargetRule(),
                ExcessiveClassesRule(),
                IdentifierTargetRule(),
            ]
        super().__init__(rules=rules)

    def validate(
        self,
        df: Any,
        target: str | None = None,
        problem_type: str | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """
        Execute target validation rules and compute Target Intelligence metadata.

        Parameters
        ----------
        df : Any
            Dataset to validate.
        target : str, optional
            Target column name.
        problem_type : str, optional
            'classification' or 'regression'. Inferred if None.
        **kwargs : Any

        Returns
        -------
        ValidationResult
        """
        # Auto-infer problem_type if not explicitly provided
        inferred_problem_type = self._infer_problem_type(df, target, problem_type)

        result = super().validate(
            df,
            target=target,
            problem_type=inferred_problem_type,
            **kwargs,
        )

        # Collect Target Intelligence metadata if target exists in DataFrame
        if isinstance(df, pd.DataFrame) and target and target in df.columns:
            series = df[target]
            non_null = series.dropna()
            n_unique = int(non_null.nunique()) if len(non_null) > 0 else 0
            n_missing = int(series.isna().sum())
            missing_pct = round(n_missing / len(df) * 100, 2) if len(df) > 0 else 0.0

            # Target Statistics & Intelligence
            intel: dict[str, Any] = {
                "target_name": target,
                "problem_type": inferred_problem_type,
                "dtype": str(series.dtype),
                "n_unique": n_unique,
                "missing_count": n_missing,
                "missing_pct": missing_pct,
            }

            if inferred_problem_type == "classification":
                counts = non_null.value_counts()
                dist = {str(k): int(v) for k, v in counts.head(10).items()}
                intel["distribution"] = dist
                is_imbalanced = any(msg.rule_id == "KML-T006" for msg in result.messages)
                intel["recommended_metric"] = "F1 Score" if is_imbalanced else "Accuracy"
                intel["recommended_stratified_split"] = True
            else:  # regression
                if pd.api.types.is_numeric_dtype(series) and len(non_null) > 0:
                    intel["min"] = float(non_null.min())
                    intel["max"] = float(non_null.max())
                    intel["mean"] = float(non_null.mean())
                    intel["std"] = float(non_null.std()) if len(non_null) > 1 else 0.0
                intel["recommended_metric"] = "RMSE"
                intel["recommended_stratified_split"] = False

            result.statistics["target_intelligence"] = intel

        # Target Health Score
        score, rating = self._compute_target_health_score(result)
        result.statistics["health_score"] = score
        result.statistics["health_rating"] = rating

        return result

    def _infer_problem_type(self, df: Any, target: str | None, problem_type: str | None) -> str:
        """Infer 'classification' or 'regression' if not specified."""
        if problem_type in ("classification", "regression"):
            return problem_type

        if isinstance(df, pd.DataFrame) and target and target in df.columns:
            series = df[target]
            if pd.api.types.is_numeric_dtype(series):
                n_unique = series.dropna().nunique()
                if n_unique > 20:
                    return "regression"
                return "classification"
            return "classification"

        return problem_type or "classification"

    def _compute_target_health_score(self, result: ValidationResult) -> tuple[int, str]:
        """Calculate Target Health Score (0–100) and Rating."""
        if any(
            msg.rule_id in ("KML-T001", "KML-T002", "KML-T003", "KML-T005", "KML-T007", "KML-T008")
            for msg in result.messages
        ):
            return 0, "★☆☆☆☆ Unusable"

        score = 100
        for msg in result.messages:
            rule_id = msg.rule_id
            if rule_id == "KML-T004" and msg.severity == "error":
                score -= 20
            elif rule_id in ("KML-T004", "KML-T006", "KML-T009", "KML-T010"):
                score -= 5

        score = max(0, min(100, score))

        if score >= 90:
            rating = "★★★★★ Excellent"
        elif score >= 70:
            rating = "★★★★☆ Good"
        elif score >= 50:
            rating = "★★★☆☆ Fair"
        else:
            rating = "★☆☆☆☆ Poor"

        return score, rating
