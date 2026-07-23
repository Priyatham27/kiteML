"""
target_rules.py — Target column validation rules (KML-T001 through KML-T010).
"""

from typing import Any

import numpy as np
import pandas as pd

from kiteml.validation.message import ValidationMessage
from kiteml.validation.rule import ValidationRule
from kiteml.validation.severity import ValidationSeverity

_ID_KEYWORDS = {"id", "uuid", "guid", "key", "index", "code", "ref", "no", "num", "number", "sku"}


class TargetSpecifiedRule(ValidationRule):
    rule_id = "KML-T001"
    name = "Target Specified Check"
    description = "Verify that a target column name parameter was provided."
    default_severity = ValidationSeverity.CRITICAL

    def check(self, df: pd.DataFrame, target: str | None = None, **kwargs: Any) -> ValidationMessage | None:
        if not target or not str(target).strip():
            return ValidationMessage(
                severity=self.default_severity,
                title="Target Column Not Specified",
                description="Target column parameter was not provided (received None or empty string).",
                suggestion="Specify a target column name (e.g., target='label').",
                rule_id=self.rule_id,
                code=self.rule_id,
            )
        return None


class TargetExistsRule(ValidationRule):
    rule_id = "KML-T002"
    name = "Target Column Existence Check"
    description = "Verify that the requested target column exists in the dataset."
    default_severity = ValidationSeverity.CRITICAL

    def check(self, df: pd.DataFrame, target: str | None = None, **kwargs: Any) -> ValidationMessage | None:
        if not isinstance(df, pd.DataFrame) or not target:
            return None

        if target not in df.columns:
            cols = [str(c) for c in df.columns[:10]]
            return ValidationMessage(
                severity=self.default_severity,
                title="Target Column Not Found",
                description=f"Target column '{target}' was not found in dataset columns.",
                suggestion=f"Verify target column name spelling. Available columns sample: {cols}",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"requested_target": target, "available_columns_sample": cols},
            )
        return None


class TargetContainsValuesRule(ValidationRule):
    rule_id = "KML-T003"
    name = "Target Non-Empty Values Check"
    description = "Verify that the target column contains non-null values."
    default_severity = ValidationSeverity.CRITICAL

    def check(self, df: pd.DataFrame, target: str | None = None, **kwargs: Any) -> ValidationMessage | None:
        if not isinstance(df, pd.DataFrame) or not target or target not in df.columns:
            return None

        non_null = df[target].dropna()
        if len(non_null) == 0:
            return ValidationMessage(
                severity=self.default_severity,
                title="Target Column Completely Empty",
                description=f"Target column '{target}' is 100% missing (all NaN).",
                suggestion="Provide a target column containing valid non-null labels/values.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"target": target, "n_rows": len(df)},
            )
        return None


class TargetMissingPercentageRule(ValidationRule):
    rule_id = "KML-T004"
    name = "Target Missing Percentage Check"
    description = "Evaluate percentage of missing values in the target column."
    default_severity = ValidationSeverity.WARNING

    def check(self, df: pd.DataFrame, target: str | None = None, **kwargs: Any) -> ValidationMessage | None:
        if not isinstance(df, pd.DataFrame) or not target or target not in df.columns:
            return None

        n_rows = len(df)
        if n_rows == 0:
            return None

        n_null = int(df[target].isna().sum())
        missing_ratio = n_null / n_rows

        if missing_ratio > 0.20:
            return ValidationMessage(
                severity=ValidationSeverity.ERROR,
                title="Excessive Target Missing Values (>20%)",
                description=f"Target '{target}' has {n_null} missing values ({missing_ratio:.1%}).",
                suggestion="Impute or drop missing target rows before training.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"missing_count": n_null, "missing_ratio": round(missing_ratio, 4)},
            )
        elif missing_ratio > 0.05:
            return ValidationMessage(
                severity=ValidationSeverity.WARNING,
                title="Target Column Contains Missing Values",
                description=f"Target '{target}' has {n_null} missing values ({missing_ratio:.1%}).",
                suggestion="Target rows with missing values will be dropped during preprocessing.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"missing_count": n_null, "missing_ratio": round(missing_ratio, 4)},
            )
        return None


class MinClassesClassificationRule(ValidationRule):
    rule_id = "KML-T005"
    name = "Classification Minimum Classes Check"
    description = "Verify that classification targets contain at least 2 distinct classes."
    default_severity = ValidationSeverity.CRITICAL

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
        n_classes = int(non_null.nunique())

        if n_classes < 2:
            return ValidationMessage(
                severity=self.default_severity,
                title="Insufficient Target Classes for Classification",
                description=f"Classification target '{target}' has only {n_classes} class(es). Minimum 2 required.",
                suggestion="Ensure the target column contains at least 2 distinct class labels.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"target": target, "n_classes": n_classes},
            )
        return None


class ClassImbalanceRule(ValidationRule):
    rule_id = "KML-T006"
    name = "Class Imbalance Check"
    description = "Detect class distribution imbalance for classification targets."
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
        n_total = len(non_null)
        if n_total < 2 or non_null.nunique() < 2:
            return None

        counts = non_null.value_counts()
        min_count = int(counts.min())
        minority_ratio = min_count / n_total
        minority_class = counts.index[-1]

        if minority_ratio < 0.02:
            return ValidationMessage(
                severity=ValidationSeverity.WARNING,
                title="Severe Class Imbalance Detected (<2%)",
                description=f"Minority class '{minority_class}' makes up only {minority_ratio:.1%} of samples ({min_count}/{n_total}).",
                suggestion="Use class weighting or resample minority class (SMOTE/oversampling).",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"minority_class": str(minority_class), "minority_ratio": round(minority_ratio, 4)},
            )
        elif minority_ratio < 0.10:
            return ValidationMessage(
                severity=ValidationSeverity.WARNING,
                title="High Class Imbalance Detected (<10%)",
                description=f"Minority class '{minority_class}' makes up {minority_ratio:.1%} of samples ({min_count}/{n_total}).",
                suggestion="Consider stratified split and PR-AUC/F1 metrics.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"minority_class": str(minority_class), "minority_ratio": round(minority_ratio, 4)},
            )
        elif minority_ratio < 0.20:
            return ValidationMessage(
                severity=ValidationSeverity.INFO,
                title="Moderate Class Imbalance Detected",
                description=f"Minority class '{minority_class}' represents {minority_ratio:.1%} of samples.",
                suggestion="Evaluate using stratified cross-validation.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"minority_class": str(minority_class), "minority_ratio": round(minority_ratio, 4)},
            )
        return None


class ConstantTargetRule(ValidationRule):
    rule_id = "KML-T007"
    name = "Constant Target Variance Check"
    description = "Verify that the target column has non-zero variance."
    default_severity = ValidationSeverity.CRITICAL

    def check(self, df: pd.DataFrame, target: str | None = None, **kwargs: Any) -> ValidationMessage | None:
        if not isinstance(df, pd.DataFrame) or not target or target not in df.columns:
            return None

        non_null = df[target].dropna()
        if len(non_null) > 0 and non_null.nunique() <= 1:
            val = non_null.iloc[0]
            return ValidationMessage(
                severity=self.default_severity,
                title="Constant Target Column",
                description=f"Target '{target}' has zero variance (constant value: '{val}'). Training is impossible.",
                suggestion="Choose a target column with varying labels/values.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"target": target, "constant_value": str(val)},
            )
        return None


class NumericRegressionTargetRule(ValidationRule):
    rule_id = "KML-T008"
    name = "Regression Target Numeric Check"
    description = "Verify that regression targets have a numeric data type."
    default_severity = ValidationSeverity.CRITICAL

    def check(
        self,
        df: pd.DataFrame,
        target: str | None = None,
        problem_type: str | None = None,
        **kwargs: Any,
    ) -> ValidationMessage | None:
        if not isinstance(df, pd.DataFrame) or not target or target not in df.columns:
            return None

        if problem_type != "regression":
            return None

        series = df[target]
        if not pd.api.types.is_numeric_dtype(series):
            return ValidationMessage(
                severity=self.default_severity,
                title="Non-Numeric Regression Target",
                description=f"Regression target '{target}' of dtype '{series.dtype}' is not numeric.",
                suggestion="Convert target to float/int dtype or switch to classification problem type.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"target": target, "dtype": str(series.dtype)},
            )
        return None


class ExcessiveClassesRule(ValidationRule):
    rule_id = "KML-T009"
    name = "Excessive Classification Classes Check"
    description = "Detect high-cardinality targets (>100 classes) for classification."
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

        n_classes = int(df[target].dropna().nunique())
        if n_classes > 100:
            return ValidationMessage(
                severity=self.default_severity,
                title="High Cardinality Classification Target (>100 Classes)",
                description=f"Classification target '{target}' contains {n_classes} unique classes.",
                suggestion="Verify if target is an identifier or aggregate rare classes before training.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"target": target, "n_classes": n_classes},
            )
        return None


class IdentifierTargetRule(ValidationRule):
    rule_id = "KML-T010"
    name = "Identifier-Like Target Check"
    description = "Warn if the target column appears to be a unique row identifier or UUID."
    default_severity = ValidationSeverity.WARNING

    def check(self, df: pd.DataFrame, target: str | None = None, **kwargs: Any) -> ValidationMessage | None:
        if not isinstance(df, pd.DataFrame) or not target or target not in df.columns:
            return None

        non_null = df[target].dropna()
        n_total = len(non_null)
        if n_total < 10:
            return None

        n_unique = int(non_null.nunique())
        unique_ratio = n_unique / n_total
        target_lower = str(target).lower().replace(" ", "_")

        is_id_name = any(kw in target_lower for kw in _ID_KEYWORDS)
        if unique_ratio > 0.95 and is_id_name:
            return ValidationMessage(
                severity=self.default_severity,
                title="Target Column Appears to be an Identifier",
                description=f"Target '{target}' is {unique_ratio:.1%} unique and matches identifier patterns.",
                suggestion="Ensure the target column is not a row index, customer ID, or UUID.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"target": target, "unique_ratio": round(unique_ratio, 4)},
            )
        return None
