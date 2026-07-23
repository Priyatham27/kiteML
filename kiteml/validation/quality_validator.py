"""
quality_validator.py — Composite Data Quality Validator for KiteML.
"""

from typing import Any

import pandas as pd

from kiteml.validation.quality_profile import QualityProfile
from kiteml.validation.rules.quality_rules import (
    ClassBalanceAnalysisRule,
    DataConsistencyRule,
    DatasetHealthScoreRule,
    DuplicateRowsAnalysisRule,
    FullyEmptyColumnsRule,
    FullyEmptyRowsRule,
    HighCorrelationRule,
    MissingValueAnalysisRule,
    NearZeroVarianceRule,
    OutlierDetectionRule,
)
from kiteml.validation.utils import get_dataframe_memory_mb
from kiteml.validation.validation_result import ValidationResult
from kiteml.validation.validator import BaseValidator


class QualityValidator(BaseValidator):
    """
    Evaluates dataset health, missing value patterns, outliers, correlations,
    near-zero variance, class balance, data consistency, and produces Dataset Health Score
    (0-100), letter grade (A+, A, B, C, Needs Attention), and QualityProfile.
    """

    description: str = "Evaluates dataset data quality, health scores, and recommendations."

    @property
    def name(self) -> str:
        return "QualityValidator"

    def __init__(self, rules: list[Any] | None = None) -> None:
        if rules is None:
            rules = [
                MissingValueAnalysisRule(),
                FullyEmptyColumnsRule(),
                FullyEmptyRowsRule(),
                OutlierDetectionRule(),
                HighCorrelationRule(),
                NearZeroVarianceRule(),
                DuplicateRowsAnalysisRule(),
                ClassBalanceAnalysisRule(),
                DataConsistencyRule(),
                DatasetHealthScoreRule(),
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
        Execute quality validation rules and compute Dataset Health Score and QualityProfile.

        Parameters
        ----------
        df : Any
            Dataset to validate.
        target : str, optional
            Target column name.
        problem_type : str, optional
            'classification' or 'regression'.
        **kwargs : Any

        Returns
        -------
        ValidationResult
        """
        result = super().validate(df, target=target, problem_type=problem_type, **kwargs)

        # Build QualityProfile if df is a valid DataFrame
        if isinstance(df, pd.DataFrame):
            score, grade, rating = self._compute_quality_health_score(result, df)
            recommendations = self._extract_recommendations(result)

            profile = QualityProfile(
                overall_score=score,
                overall_grade=grade,
                health_rating=rating,
                missing_summary=self._build_missing_summary(df),
                duplicate_summary=self._build_duplicate_summary(df),
                outlier_summary=self._build_outlier_summary(df, target),
                correlation_summary=self._build_correlation_summary(df, target),
                variance_summary=self._build_variance_summary(df, target),
                balance_summary=self._build_balance_summary(df, target, problem_type),
                consistency_summary=self._build_consistency_summary(df, target),
                memory_summary=self._build_memory_summary(df),
                recommendations=recommendations,
            )

            result.statistics["quality_profile"] = profile.to_dict()
            result.statistics["health_score"] = score
            result.statistics["health_grade"] = grade
            result.statistics["health_rating"] = rating
            result.statistics["recommendations"] = recommendations

        return result

    def _compute_quality_health_score(
        self,
        result: ValidationResult,
        df: pd.DataFrame,
    ) -> tuple[int, str, str]:
        """Calculate Dataset Health Score (0-100), Grade, and Star Rating."""
        score = 100

        # Check for critical/error rules
        for msg in result.messages:
            rule_id = msg.rule_id
            if rule_id == "KML-Q002":  # Fully empty columns
                score -= 10
            elif rule_id == "KML-Q001":  # Missing values
                if msg.severity in ("critical", "error"):
                    score -= 10
                else:
                    score -= 5
            elif rule_id == "KML-Q007":  # Duplicate rows
                score -= 3
            elif rule_id in ("KML-Q003", "KML-Q006"):  # Empty rows / Near zero variance
                score -= 4
            elif rule_id == "KML-Q005":  # High correlation
                score -= 2
            elif rule_id == "KML-Q008":  # Severe imbalance
                score -= 8
            elif rule_id in ("KML-Q004", "KML-Q009"):  # Outliers / Consistency issues
                score -= 2

        score = max(0, min(100, score))

        if score >= 95:
            grade = "A+"
            rating = "★★★★★ Excellent"
        elif score >= 90:
            grade = "A"
            rating = "★★★★★ Excellent"
        elif score >= 80:
            grade = "B"
            rating = "★★★★☆ Good"
        elif score >= 70:
            grade = "C"
            rating = "★★★☆☆ Fair"
        else:
            grade = "Needs Attention"
            rating = "★☆☆☆☆ Poor"

        return score, grade, rating

    def _extract_recommendations(self, result: ValidationResult) -> list[str]:
        """Collect unique actionable recommendations from validation messages."""
        recs: list[str] = []
        for msg in result.messages:
            if msg.suggestion and msg.suggestion not in recs:
                recs.append(msg.suggestion)
        return recs

    def _build_missing_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        n_total = df.size
        n_missing = int(df.isna().sum().sum())
        ratio = round(n_missing / n_total * 100, 2) if n_total > 0 else 0.0
        empty_cols = [str(c) for c in df.columns if df[c].isna().all()]
        return {
            "missing_cells": n_missing,
            "missing_percentage": ratio,
            "fully_empty_columns": empty_cols,
        }

    def _build_duplicate_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        n_dups = int(df.duplicated().sum()) if len(df) > 0 else 0
        ratio = round(n_dups / len(df) * 100, 2) if len(df) > 0 else 0.0
        return {"duplicate_rows": n_dups, "duplicate_percentage": ratio}

    def _build_outlier_summary(self, df: pd.DataFrame, target: str | None) -> dict[str, Any]:
        outlier_cols: dict[str, int] = {}
        for c in df.columns:
            if target and str(c) == str(target):
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                non_null = df[c].dropna()
                if len(non_null) > 10:
                    q25, q75 = pd.Series(non_null).quantile([0.25, 0.75])
                    iqr = q75 - q25
                    if iqr > 0:
                        count = int(((non_null < q25 - 1.5 * iqr) | (non_null > q75 + 1.5 * iqr)).sum())
                        if count > 0:
                            outlier_cols[str(c)] = count
        return {"method": "IQR (1.5x)", "outlier_columns": outlier_cols}

    def _build_correlation_summary(self, df: pd.DataFrame, target: str | None) -> dict[str, Any]:
        cols = [c for c in df.columns if (not target or str(c) != str(target)) and pd.api.types.is_numeric_dtype(df[c])]
        high_pairs: list[dict[str, Any]] = []
        if len(cols) >= 2:
            corr = df[cols].corr(numeric_only=True).abs()
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    c1, c2 = cols[i], cols[j]
                    val = corr.loc[c1, c2]
                    if not pd.isna(val) and val > 0.90:
                        high_pairs.append(
                            {"feature_1": str(c1), "feature_2": str(c2), "correlation": round(float(val), 3)}
                        )
        return {"threshold": 0.90, "high_correlation_pairs": high_pairs}

    def _build_variance_summary(self, df: pd.DataFrame, target: str | None) -> dict[str, Any]:
        nzv: list[str] = []
        for c in df.columns:
            if target and str(c) == str(target):
                continue
            non_null = df[c].dropna()
            if len(non_null) > 10:
                top_freq = non_null.value_counts().iloc[0]
                if top_freq / len(non_null) > 0.99 and non_null.nunique() > 1:
                    nzv.append(str(c))
        return {"near_zero_variance_columns": nzv}

    def _build_balance_summary(self, df: pd.DataFrame, target: str | None, problem_type: str | None) -> dict[str, Any]:
        if not target or str(target) not in df.columns or problem_type != "classification":
            return {"status": "N/A"}

        non_null = df[str(target)].dropna()
        if len(non_null) < 2 or non_null.nunique() < 2:
            return {"status": "Insufficient Target Classes"}

        counts = non_null.value_counts()
        maj_ratio = round(counts.iloc[0] / len(non_null) * 100, 2)
        min_ratio = round(counts.iloc[-1] / len(non_null) * 100, 2)

        status = "Imbalanced" if min_ratio < 10.0 else "Healthy"

        return {
            "status": status,
            "majority_percentage": maj_ratio,
            "minority_percentage": min_ratio,
        }

    def _build_consistency_summary(self, df: pd.DataFrame, target: str | None) -> dict[str, Any]:
        pseudo_nulls: dict[str, int] = {}
        for c in df.columns:
            if target and str(c) == str(target):
                continue
            series = df[c]
            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                non_null = series.dropna().astype(str)
                count = sum(
                    1
                    for val in non_null
                    if not val.strip() or val.strip().lower() in {"na", "n/a", "null", "none", "-", "?"}
                )
                if count > 0:
                    pseudo_nulls[str(c)] = count
        return {"pseudo_null_columns": pseudo_nulls}

    def _build_memory_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        mem_mb = get_dataframe_memory_mb(df)
        col_sizes = {str(c): round(df[c].memory_usage(deep=True) / (1024 * 1024), 3) for c in df.columns}
        largest_cols = dict(sorted(col_sizes.items(), key=lambda item: item[1], reverse=True)[:5])
        return {"total_memory_mb": mem_mb, "largest_columns_mb": largest_cols}
