"""
dataset_validator.py — Composite Dataset Validator for KiteML.
"""

from typing import Any

import pandas as pd

from kiteml.validation.rules.dataset_rules import (
    DatasetExistsRule,
    DatasetIsDataFrameRule,
    DatasetNotEmptyRule,
    DuplicateColumnsRule,
    DuplicateRowsRule,
    EmptyColumnNamesRule,
    EmptyRowsRule,
    InfiniteValuesRule,
    MinColumnsRule,
    MinRowsRule,
    ReservedColumnNamesRule,
    UnsupportedValuesRule,
)
from kiteml.validation.validation_result import ValidationResult
from kiteml.validation.validator import BaseValidator


class DatasetValidator(BaseValidator):
    """
    Validates structural integrity, shape, headers, rows, and cell types of a dataset.

    Calculates an overall Dataset Health Score (0–100) and Health Rating.
    """

    description: str = "Validates overall dataset structural integrity and cell health."

    @property
    def name(self) -> str:
        return "DatasetValidator"

    def __init__(self, rules: list[Any] | None = None) -> None:
        if rules is None:
            rules = [
                DatasetExistsRule(),
                DatasetIsDataFrameRule(),
                DatasetNotEmptyRule(),
                MinRowsRule(),
                MinColumnsRule(),
                DuplicateColumnsRule(),
                EmptyColumnNamesRule(),
                ReservedColumnNamesRule(),
                DuplicateRowsRule(),
                EmptyRowsRule(),
                InfiniteValuesRule(),
                UnsupportedValuesRule(),
            ]
        super().__init__(rules=rules)

    def validate(self, df: Any, **kwargs: Any) -> ValidationResult:
        """
        Execute dataset validation rules and compute dataset health score.

        Parameters
        ----------
        df : Any
            Dataset to validate (expected pandas.DataFrame).
        **kwargs : Any

        Returns
        -------
        ValidationResult
        """
        result = super().validate(df, **kwargs)

        # Additional statistics if df is a valid DataFrame
        if isinstance(df, pd.DataFrame) and not df.empty and len(df.columns) > 0:
            try:
                n_dup_rows = int(df.duplicated().sum())
            except Exception:
                n_dup_rows = 0

            try:
                n_empty_rows = int(df.isna().all(axis=1).sum())
            except Exception:
                n_empty_rows = 0

            try:
                n_missing_cells = int(df.isna().sum().sum())
            except Exception:
                n_missing_cells = 0

            result.statistics["duplicate_rows"] = n_dup_rows
            result.statistics["empty_rows"] = n_empty_rows
            result.statistics["missing_cells"] = n_missing_cells

        # Compute Dataset Health Score
        score, rating = self._compute_health_score(result)
        result.statistics["health_score"] = score
        result.statistics["health_rating"] = rating

        return result

    def _compute_health_score(self, result: ValidationResult) -> tuple[int, str]:
        """
        Calculate Dataset Health Score (0–100) based on detected rule violations.

        Deductions:
        - Critical structural failure: automatic score = 0
        - KML-D011 (Infinite values): -20
        - KML-D006 (Duplicate columns): -10
        - KML-D007 (Empty column names): -10
        - KML-D012 (Unsupported values): -10
        - KML-D010 (Empty rows): -5
        - KML-D009 (Duplicate rows): -2
        - KML-D008 (Reserved names): -1
        """
        if any(msg.rule_id in ("KML-D001", "KML-D002", "KML-D003", "KML-D004", "KML-D005") for msg in result.messages):
            return 0, "★☆☆☆☆ Unusable"

        score = 100
        for msg in result.messages:
            rule_id = msg.rule_id
            if rule_id == "KML-D011":
                score -= 20
            elif rule_id in ("KML-D006", "KML-D007", "KML-D012"):
                score -= 10
            elif rule_id == "KML-D010":
                score -= 5
            elif rule_id == "KML-D009":
                score -= 2
            elif rule_id == "KML-D008":
                score -= 1

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
