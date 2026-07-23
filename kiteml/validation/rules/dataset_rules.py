"""
dataset_rules.py — Dataset-level validation rules (KML-D001 through KML-D012).
"""

from typing import Any

import numpy as np
import pandas as pd

from kiteml.validation.message import ValidationMessage
from kiteml.validation.rule import ValidationRule
from kiteml.validation.severity import ValidationSeverity


class DatasetExistsRule(ValidationRule):
    rule_id = "KML-D001"
    name = "Dataset Exists Check"
    description = "Verify that a dataset object was provided (not None)."
    default_severity = ValidationSeverity.CRITICAL

    def check(self, df: Any, **kwargs: Any) -> ValidationMessage | None:
        if df is None:
            return ValidationMessage(
                severity=self.default_severity,
                title="Dataset Missing",
                description="No dataset object was provided (received None).",
                suggestion="Pass a valid pandas DataFrame to the validator.",
                rule_id=self.rule_id,
                code=self.rule_id,
            )
        return None


class DatasetIsDataFrameRule(ValidationRule):
    rule_id = "KML-D002"
    name = "Dataset DataFrame Type Check"
    description = "Verify that dataset is a pandas DataFrame."
    default_severity = ValidationSeverity.CRITICAL

    def check(self, df: Any, **kwargs: Any) -> ValidationMessage | None:
        if df is not None and not isinstance(df, pd.DataFrame):
            return ValidationMessage(
                severity=self.default_severity,
                title="Dataset is Not a DataFrame",
                description=f"Expected pandas.DataFrame, received '{type(df).__name__}'.",
                suggestion="Convert your data structure to pandas.DataFrame before validation.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"type": type(df).__name__},
            )
        return None


class DatasetNotEmptyRule(ValidationRule):
    rule_id = "KML-D003"
    name = "Dataset Not Empty Check"
    description = "Verify that the DataFrame is not completely empty."
    default_severity = ValidationSeverity.CRITICAL

    def check(self, df: pd.DataFrame, **kwargs: Any) -> ValidationMessage | None:
        if isinstance(df, pd.DataFrame) and df.empty:
            return ValidationMessage(
                severity=self.default_severity,
                title="Dataset Empty",
                description="Dataset contains 0 rows or 0 columns.",
                suggestion="Provide a dataset containing at least 1 row and 1 column.",
                rule_id=self.rule_id,
                code=self.rule_id,
            )
        return None


class MinRowsRule(ValidationRule):
    rule_id = "KML-D004"
    name = "Minimum Rows Check"
    description = "Verify that the dataset contains at least 1 row."
    default_severity = ValidationSeverity.CRITICAL

    def check(self, df: pd.DataFrame, **kwargs: Any) -> ValidationMessage | None:
        if isinstance(df, pd.DataFrame) and len(df) == 0:
            return ValidationMessage(
                severity=self.default_severity,
                title="Insufficient Rows",
                description="Dataset contains 0 rows.",
                suggestion="Provide a dataset with at least 1 row.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"n_rows": 0},
            )
        return None


class MinColumnsRule(ValidationRule):
    rule_id = "KML-D005"
    name = "Minimum Columns Check"
    description = "Verify that the dataset contains at least 1 column."
    default_severity = ValidationSeverity.CRITICAL

    def check(self, df: pd.DataFrame, **kwargs: Any) -> ValidationMessage | None:
        if isinstance(df, pd.DataFrame) and len(df.columns) == 0:
            return ValidationMessage(
                severity=self.default_severity,
                title="Insufficient Columns",
                description="Dataset contains 0 feature columns.",
                suggestion="Provide a dataset with at least 1 feature column.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"n_cols": 0},
            )
        return None


class DuplicateColumnsRule(ValidationRule):
    rule_id = "KML-D006"
    name = "Duplicate Column Names Check"
    description = "Detect duplicate column names in the dataset header."
    default_severity = ValidationSeverity.ERROR

    def check(self, df: pd.DataFrame, **kwargs: Any) -> ValidationMessage | None:
        if not isinstance(df, pd.DataFrame) or len(df.columns) == 0:
            return None

        cols = [str(c) for c in df.columns]
        dups = [c for c in set(cols) if cols.count(c) > 1]
        if dups:
            return ValidationMessage(
                severity=self.default_severity,
                title="Duplicate Column Names Detected",
                description=f"Found {len(dups)} duplicate column name(s): {dups}",
                suggestion="Ensure all column names are unique using df.columns.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"duplicates": dups},
            )
        return None


class EmptyColumnNamesRule(ValidationRule):
    rule_id = "KML-D007"
    name = "Blank Column Names Check"
    description = "Detect empty, whitespace-only, or 'Unnamed' column names."
    default_severity = ValidationSeverity.ERROR

    def check(self, df: pd.DataFrame, **kwargs: Any) -> ValidationMessage | None:
        if not isinstance(df, pd.DataFrame) or len(df.columns) == 0:
            return None

        blank_cols: list[str] = []
        for col in df.columns:
            s_col = str(col).strip()
            if not s_col or col is None or s_col.startswith("Unnamed:"):
                blank_cols.append(str(col))

        if blank_cols:
            return ValidationMessage(
                severity=self.default_severity,
                title="Empty or Blank Column Names",
                description=f"Found {len(blank_cols)} empty/unnamed column name(s): {blank_cols}",
                suggestion="Assign meaningful string names to all columns.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"blank_columns": blank_cols},
            )
        return None


class ReservedColumnNamesRule(ValidationRule):
    rule_id = "KML-D008"
    name = "Reserved Column Names Check"
    description = "Check for internal reserved KiteML column names."
    default_severity = ValidationSeverity.WARNING

    RESERVED_NAMES = {"_kiteml_", "_prediction_", "_probability_", "_row_id_", "_target_"}

    def check(self, df: pd.DataFrame, **kwargs: Any) -> ValidationMessage | None:
        if not isinstance(df, pd.DataFrame) or len(df.columns) == 0:
            return None

        found_reserved = [str(c) for c in df.columns if str(c) in self.RESERVED_NAMES]
        if found_reserved:
            return ValidationMessage(
                severity=self.default_severity,
                title="Reserved Column Names Detected",
                description=f"Dataset contains internal reserved column name(s): {found_reserved}",
                suggestion="Rename reserved columns to avoid conflicts with KiteML internals.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"reserved_columns": found_reserved},
            )
        return None


class DuplicateRowsRule(ValidationRule):
    rule_id = "KML-D009"
    name = "Duplicate Rows Check"
    description = "Detect duplicate rows in the dataset."
    default_severity = ValidationSeverity.WARNING

    def check(self, df: pd.DataFrame, **kwargs: Any) -> ValidationMessage | None:
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            return None

        try:
            n_dups = int(df.duplicated().sum())
            if n_dups > 0:
                ratio = n_dups / len(df)
                return ValidationMessage(
                    severity=self.default_severity,
                    title="Duplicate Rows Detected",
                    description=f"{n_dups} duplicate row(s) found ({ratio:.1%}).",
                    suggestion="Consider removing duplicate rows using df.drop_duplicates().",
                    rule_id=self.rule_id,
                    code=self.rule_id,
                    context={"duplicate_count": n_dups, "duplicate_ratio": round(ratio, 4)},
                )
        except Exception:
            pass
        return None


class EmptyRowsRule(ValidationRule):
    rule_id = "KML-D010"
    name = "Completely Empty Rows Check"
    description = "Detect rows where all feature values are missing (NaN)."
    default_severity = ValidationSeverity.WARNING

    def check(self, df: pd.DataFrame, **kwargs: Any) -> ValidationMessage | None:
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            return None

        try:
            n_empty = int(df.isna().all(axis=1).sum())
            if n_empty > 0:
                ratio = n_empty / len(df)
                return ValidationMessage(
                    severity=self.default_severity,
                    title="Completely Empty Rows Detected",
                    description=f"{n_empty} row(s) are completely empty (all NaN).",
                    suggestion="Remove empty rows using df.dropna(how='all').",
                    rule_id=self.rule_id,
                    code=self.rule_id,
                    context={"empty_row_count": n_empty, "empty_row_ratio": round(ratio, 4)},
                )
        except Exception:
            pass
        return None


class InfiniteValuesRule(ValidationRule):
    rule_id = "KML-D011"
    name = "Infinite Values Check"
    description = "Detect numerical inf or -inf values in dataset."
    default_severity = ValidationSeverity.ERROR

    def check(self, df: pd.DataFrame, **kwargs: Any) -> ValidationMessage | None:
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            return None

        inf_cols: dict[str, int] = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col]
            try:
                inf_count = int(np.isinf(series).sum())
                if inf_count > 0:
                    inf_cols[str(col)] = inf_count
            except Exception:
                continue

        if inf_cols:
            total_infs = sum(inf_cols.values())
            return ValidationMessage(
                severity=self.default_severity,
                title="Infinite Values Detected",
                description=f"Found {total_infs} infinite value(s) in {len(inf_cols)} numeric column(s): {list(inf_cols.keys())}",
                suggestion="Replace infinite values with NaN or finite numerical bounds.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"infinite_columns": inf_cols, "total_inf_count": total_infs},
            )
        return None


class UnsupportedValuesRule(ValidationRule):
    rule_id = "KML-D012"
    name = "Unsupported Cell Objects Check"
    description = "Detect complex numbers or nested containers (dict, list, set) in dataset cells."
    default_severity = ValidationSeverity.ERROR

    def check(self, df: pd.DataFrame, **kwargs: Any) -> ValidationMessage | None:
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            return None

        unsupported_cols: dict[str, str] = {}
        for col in df.columns:
            series = df[col]
            if pd.api.types.is_object_dtype(series):
                sample = series.dropna().head(100)
                for item in sample:
                    if isinstance(item, (dict, list, set, tuple, complex)):
                        unsupported_cols[str(col)] = type(item).__name__
                        break

        if unsupported_cols:
            return ValidationMessage(
                severity=self.default_severity,
                title="Unsupported Cell Objects Detected",
                description=f"Found unhandled Python objects in columns: {unsupported_cols}",
                suggestion="Convert nested objects (dict, list, set, complex) into primitive scalars or string representations.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"unsupported_columns": unsupported_cols},
            )
        return None
