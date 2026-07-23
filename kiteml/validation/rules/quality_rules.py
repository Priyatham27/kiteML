"""
quality_rules.py — Data quality validation rules (KML-Q001 through KML-Q010).
"""

from typing import Any

import numpy as np
import pandas as pd

from kiteml.validation.message import ValidationMessage
from kiteml.validation.rule import ValidationRule
from kiteml.validation.severity import ValidationSeverity

_NULL_STRINGS = {"na", "n/a", "null", "none", "-", "nan", "unknown", "?"}


def _get_feature_cols(df: pd.DataFrame, target: str | None = None) -> list[str]:
    """Return feature column names excluding target if specified."""
    if not isinstance(df, pd.DataFrame):
        return []
    if target and target in df.columns:
        return [str(c) for c in df.columns if str(c) != str(target)]
    return [str(c) for c in df.columns]


class MissingValueAnalysisRule(ValidationRule):
    rule_id = "KML-Q001"
    name = "Dataset Missing Value Analysis Check"
    description = "Evaluate overall dataset missing value percentage."
    default_severity = ValidationSeverity.WARNING

    def check(self, df: pd.DataFrame, target: str | None = None, **kwargs: Any) -> ValidationMessage | None:
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            return None

        total_cells = df.size
        if total_cells == 0:
            return None

        missing_cells = int(df.isna().sum().sum())
        missing_ratio = missing_cells / total_cells

        if missing_ratio > 0.60:
            return ValidationMessage(
                severity=ValidationSeverity.CRITICAL,
                title="Critical Missing Values (>60%)",
                description=f"Dataset is {missing_ratio:.1%} missing overall ({missing_cells}/{total_cells} cells).",
                suggestion="Dataset has severe missingness. Impute or acquire a cleaner data source.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"missing_cells": missing_cells, "missing_ratio": round(missing_ratio, 4)},
            )
        elif missing_ratio > 0.30:
            return ValidationMessage(
                severity=ValidationSeverity.ERROR,
                title="High Dataset Missing Values (>30%)",
                description=f"Dataset is {missing_ratio:.1%} missing overall ({missing_cells}/{total_cells} cells).",
                suggestion="Impute missing cells using median/mode or drop highly incomplete columns.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"missing_cells": missing_cells, "missing_ratio": round(missing_ratio, 4)},
            )
        elif missing_ratio >= 0.05:
            return ValidationMessage(
                severity=ValidationSeverity.WARNING,
                title="Dataset Contains Missing Values",
                description=f"Dataset is {missing_ratio:.1%} missing overall ({missing_cells}/{total_cells} cells).",
                suggestion="Impute missing cells before training models.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"missing_cells": missing_cells, "missing_ratio": round(missing_ratio, 4)},
            )
        return None


class FullyEmptyColumnsRule(ValidationRule):
    rule_id = "KML-Q002"
    name = "Fully Empty Columns Check"
    description = "Detect columns where 100% of cells are missing."
    default_severity = ValidationSeverity.ERROR

    def check(self, df: pd.DataFrame, target: str | None = None, **kwargs: Any) -> ValidationMessage | None:
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            return None

        feature_cols = _get_feature_cols(df, target)
        empty_cols: list[str] = []
        for col in feature_cols:
            if df[col].isna().all():
                empty_cols.append(col)

        if empty_cols:
            return ValidationMessage(
                severity=self.default_severity,
                title="Fully Empty Columns Detected",
                description=f"Found {len(empty_cols)} column(s) with 100% missing values: {empty_cols}",
                suggestion="Remove fully empty columns before preprocessing.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"empty_columns": empty_cols},
            )
        return None


class FullyEmptyRowsRule(ValidationRule):
    rule_id = "KML-Q003"
    name = "Fully Empty Rows Check"
    description = "Detect rows where 100% of feature values are missing."
    default_severity = ValidationSeverity.WARNING

    def check(self, df: pd.DataFrame, target: str | None = None, **kwargs: Any) -> ValidationMessage | None:
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            return None

        empty_row_count = int(df.isna().all(axis=1).sum())
        if empty_row_count > 0:
            ratio = empty_row_count / len(df)
            return ValidationMessage(
                severity=self.default_severity,
                title="Fully Empty Rows Detected",
                description=f"Found {empty_row_count} completely empty row(s) ({ratio:.1%}).",
                suggestion="Remove completely empty rows from the dataset.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"empty_row_count": empty_row_count, "empty_row_ratio": round(ratio, 4)},
            )
        return None


class OutlierDetectionRule(ValidationRule):
    rule_id = "KML-Q004"
    name = "Statistical Outlier Detection Check"
    description = "Detect numeric features with >5% statistical outliers using IQR or Z-score."
    default_severity = ValidationSeverity.WARNING

    def check(self, df: pd.DataFrame, target: str | None = None, **kwargs: Any) -> ValidationMessage | None:
        if not isinstance(df, pd.DataFrame) or len(df) < 10:
            return None

        feature_cols = _get_feature_cols(df, target)
        outlier_cols: dict[str, float] = {}

        for col in feature_cols:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                non_null = series.dropna()
                if len(non_null) > 10:
                    q25, q75 = np.percentile(non_null, [25, 75])
                    iqr = q75 - q25
                    if iqr > 0:
                        lower_bound = q25 - 1.5 * iqr
                        upper_bound = q75 + 1.5 * iqr
                        outliers = ((non_null < lower_bound) | (non_null > upper_bound)).sum()
                    else:
                        mean = non_null.mean()
                        std = non_null.std()
                        outliers = (np.abs(non_null - mean) > 3 * std).sum() if std > 0 else 0
                    ratio = float(outliers / len(non_null))
                    if ratio > 0.05:
                        outlier_cols[col] = round(ratio * 100, 1)

        if outlier_cols:
            return ValidationMessage(
                severity=self.default_severity,
                title="Statistical Outliers Detected (>5%)",
                description=f"Found numeric feature(s) with >5% outliers: {outlier_cols}",
                suggestion="Review outlier distributions or consider winsorization / capping.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"outlier_features": outlier_cols},
            )
        return None


class HighCorrelationRule(ValidationRule):
    rule_id = "KML-Q005"
    name = "High Feature Correlation Check"
    description = "Detect pairs of numeric features with correlation |r| > 0.90."
    default_severity = ValidationSeverity.WARNING

    def check(self, df: pd.DataFrame, target: str | None = None, **kwargs: Any) -> ValidationMessage | None:
        if not isinstance(df, pd.DataFrame) or len(df) < 5:
            return None

        feature_cols = [c for c in _get_feature_cols(df, target) if pd.api.types.is_numeric_dtype(df[c])]
        if len(feature_cols) < 2:
            return None

        corr_matrix = df[feature_cols].corr(numeric_only=True).abs()
        high_corr_pairs: list[dict[str, Any]] = []

        for i in range(len(feature_cols)):
            for j in range(i + 1, len(feature_cols)):
                col1, col2 = feature_cols[i], feature_cols[j]
                val = corr_matrix.loc[col1, col2]
                if not np.isnan(val) and val > 0.90:
                    high_corr_pairs.append({"feature_1": col1, "feature_2": col2, "correlation": round(float(val), 3)})

        if high_corr_pairs:
            return ValidationMessage(
                severity=self.default_severity,
                title="High Feature Correlation Detected (>0.90)",
                description=f"Found {len(high_corr_pairs)} highly correlated feature pair(s): {high_corr_pairs}",
                suggestion="Consider removing one feature from each pair to mitigate multicollinearity.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"high_correlation_pairs": high_corr_pairs},
            )
        return None


class NearZeroVarianceRule(ValidationRule):
    rule_id = "KML-Q006"
    name = "Near-Zero Variance Check"
    description = "Detect feature columns where a single value covers >99% of samples."
    default_severity = ValidationSeverity.WARNING

    def check(self, df: pd.DataFrame, target: str | None = None, **kwargs: Any) -> ValidationMessage | None:
        if not isinstance(df, pd.DataFrame) or len(df) < 10:
            return None

        feature_cols = _get_feature_cols(df, target)
        nzv_cols: list[str] = []

        for col in feature_cols:
            non_null = df[col].dropna()
            if len(non_null) > 10:
                top_freq = non_null.value_counts().iloc[0]
                ratio = top_freq / len(non_null)
                if ratio > 0.99 and non_null.nunique() > 1:
                    nzv_cols.append(col)

        if nzv_cols:
            return ValidationMessage(
                severity=self.default_severity,
                title="Near-Zero Variance Feature Detected (>99% Single Value)",
                description=f"Found feature(s) dominated by a single value (>99%): {nzv_cols}",
                suggestion="Remove near-constant features carrying minimal signal.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"near_zero_variance_features": nzv_cols},
            )
        return None


class DuplicateRowsAnalysisRule(ValidationRule):
    rule_id = "KML-Q007"
    name = "Duplicate Rows Analysis Check"
    description = "Detect duplicate rows in the dataset."
    default_severity = ValidationSeverity.WARNING

    def check(self, df: pd.DataFrame, target: str | None = None, **kwargs: Any) -> ValidationMessage | None:
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            return None

        n_dups = int(df.duplicated().sum())
        if n_dups > 0:
            dup_ratio = n_dups / len(df)
            return ValidationMessage(
                severity=self.default_severity,
                title="Duplicate Dataset Rows Detected",
                description=f"Found {n_dups} duplicate row(s) ({dup_ratio:.1%}).",
                suggestion="Deduplicate dataset rows before splitting and training.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"duplicate_rows": n_dups, "duplicate_ratio": round(dup_ratio, 4)},
            )
        return None


class ClassBalanceAnalysisRule(ValidationRule):
    rule_id = "KML-Q008"
    name = "Class Balance Quality Check"
    description = "Evaluate class distribution balance for classification targets."
    default_severity = ValidationSeverity.WARNING

    def check(
        self,
        df: pd.DataFrame,
        target: str | None = None,
        problem_type: str | None = None,
        **kwargs: Any,
    ) -> ValidationMessage | None:
        if not isinstance(df, pd.DataFrame) or not target or target not in df.columns:
            return None

        if problem_type != "classification":
            return None

        non_null = df[target].dropna()
        if len(non_null) < 5 or non_null.nunique() < 2:
            return None

        counts = non_null.value_counts()
        majority_ratio = counts.iloc[0] / len(non_null)
        minority_ratio = counts.iloc[-1] / len(non_null)

        if minority_ratio < 0.10 or majority_ratio > 0.90:
            return ValidationMessage(
                severity=self.default_severity,
                title="Imbalanced Classification Target",
                description=f"Majority class represents {majority_ratio:.1%}, minority represents {minority_ratio:.1%}.",
                suggestion="Use stratified cross-validation split, SMOTE, or class weights.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"majority_ratio": round(majority_ratio, 4), "minority_ratio": round(minority_ratio, 4)},
            )
        return None


class DataConsistencyRule(ValidationRule):
    rule_id = "KML-Q009"
    name = "Data Consistency String Check"
    description = "Detect whitespace-only strings or string pseudo-nulls ('NA', 'N/A', 'null', 'None', '-')."
    default_severity = ValidationSeverity.WARNING

    def check(self, df: pd.DataFrame, target: str | None = None, **kwargs: Any) -> ValidationMessage | None:
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            return None

        feature_cols = _get_feature_cols(df, target)
        inconsistent: dict[str, int] = {}

        for col in feature_cols:
            series = df[col]
            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                non_null = series.dropna().astype(str)
                count = 0
                for val in non_null:
                    s_val = val.strip().lower()
                    if not s_val or s_val in _NULL_STRINGS:
                        count += 1
                if count > 0:
                    inconsistent[col] = count

        if inconsistent:
            return ValidationMessage(
                severity=self.default_severity,
                title="String Pseudo-Null Consistency Issues",
                description=f"Found string pseudo-nulls ('N/A', 'null', 'None', ' ') in features: {inconsistent}",
                suggestion="Normalize pseudo-null string values to actual NaN missing values.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"inconsistent_features": inconsistent},
            )
        return None


class DatasetHealthScoreRule(ValidationRule):
    rule_id = "KML-Q010"
    name = "Dataset Health Score Summary Check"
    description = "Summary rule indicating overall dataset quality score."
    default_severity = ValidationSeverity.INFO

    def check(self, df: pd.DataFrame, **kwargs: Any) -> ValidationMessage | None:
        # Informational rule evaluated by QualityValidator orchestrator
        return None
