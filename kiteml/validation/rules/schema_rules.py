"""
schema_rules.py — Schema validation rules (KML-S001 through KML-S012).
"""

from typing import Any

import numpy as np
import pandas as pd

from kiteml.validation.message import ValidationMessage
from kiteml.validation.rule import ValidationRule
from kiteml.validation.severity import ValidationSeverity

_ID_KEYWORDS = {"id", "uuid", "guid", "key", "index", "code", "ref", "no", "num", "number", "sku"}


def _get_feature_cols(df: pd.DataFrame, target: str | None = None) -> list[str]:
    """Return feature column names excluding target if specified."""
    if not isinstance(df, pd.DataFrame):
        return []
    if target and target in df.columns:
        return [str(c) for c in df.columns if str(c) != str(target)]
    return [str(c) for c in df.columns]


class EmptyFeatureNameRule(ValidationRule):
    rule_id = "KML-S001"
    name = "Empty Feature Name Check"
    description = "Detect empty, blank, or 'Unnamed:' feature column names."
    default_severity = ValidationSeverity.ERROR

    def check(self, df: pd.DataFrame, target: str | None = None, **kwargs: Any) -> ValidationMessage | None:
        feature_cols = _get_feature_cols(df, target)
        if not feature_cols:
            return None

        blank_cols: list[str] = []
        for col in feature_cols:
            s_col = col.strip()
            if not s_col or s_col.startswith("Unnamed:"):
                blank_cols.append(col)

        if blank_cols:
            return ValidationMessage(
                severity=self.default_severity,
                title="Empty or Blank Feature Name Detected",
                description=f"Found {len(blank_cols)} blank/unnamed feature name(s): {blank_cols}",
                suggestion="Assign descriptive names to feature columns.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"blank_features": blank_cols},
            )
        return None


class DuplicateFeatureNameRule(ValidationRule):
    rule_id = "KML-S002"
    name = "Duplicate Feature Name Check"
    description = "Detect duplicate feature column names."
    default_severity = ValidationSeverity.ERROR

    def check(self, df: pd.DataFrame, target: str | None = None, **kwargs: Any) -> ValidationMessage | None:
        feature_cols = _get_feature_cols(df, target)
        if not feature_cols:
            return None

        dups = [c for c in set(feature_cols) if feature_cols.count(c) > 1]
        if dups:
            return ValidationMessage(
                severity=self.default_severity,
                title="Duplicate Feature Name Detected",
                description=f"Found duplicate feature column name(s): {dups}",
                suggestion="Rename duplicate feature columns.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"duplicate_features": dups},
            )
        return None


class UnsupportedDatatypeRule(ValidationRule):
    rule_id = "KML-S003"
    name = "Unsupported Feature Datatype Check"
    description = "Detect unhandled complex numbers or nested containers in feature cells."
    default_severity = ValidationSeverity.ERROR

    def check(self, df: pd.DataFrame, target: str | None = None, **kwargs: Any) -> ValidationMessage | None:
        feature_cols = _get_feature_cols(df, target)
        if not feature_cols:
            return None

        unsupported: dict[str, str] = {}
        for col in feature_cols:
            series = df[col]
            if pd.api.types.is_object_dtype(series):
                sample = series.dropna().head(100)
                for item in sample:
                    if isinstance(item, (dict, list, set, tuple, complex)):
                        unsupported[col] = type(item).__name__
                        break

        if unsupported:
            return ValidationMessage(
                severity=self.default_severity,
                title="Unsupported Datatype in Features",
                description=f"Found unsupported object types in features: {unsupported}",
                suggestion="Convert nested objects to string or flat numeric representations.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"unsupported_features": unsupported},
            )
        return None


class ConstantFeatureRule(ValidationRule):
    rule_id = "KML-S004"
    name = "Constant Feature Check"
    description = "Detect features with zero variance (single unique value)."
    default_severity = ValidationSeverity.WARNING

    def check(self, df: pd.DataFrame, target: str | None = None, **kwargs: Any) -> ValidationMessage | None:
        feature_cols = _get_feature_cols(df, target)
        if not feature_cols:
            return None

        constant_features: list[str] = []
        for col in feature_cols:
            non_null = df[col].dropna()
            if len(non_null) > 0 and non_null.nunique() <= 1:
                constant_features.append(col)

        if constant_features:
            return ValidationMessage(
                severity=self.default_severity,
                title="Constant Feature Detected",
                description=f"Found {len(constant_features)} constant feature(s) carrying 0 information: {constant_features}",
                suggestion="Remove constant features before training.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"constant_features": constant_features},
            )
        return None


class HighCardinalityFeatureRule(ValidationRule):
    rule_id = "KML-S005"
    name = "High Cardinality Feature Check"
    description = "Detect high-cardinality categorical features (>50% unique ratio)."
    default_severity = ValidationSeverity.WARNING

    def check(self, df: pd.DataFrame, target: str | None = None, **kwargs: Any) -> ValidationMessage | None:
        feature_cols = _get_feature_cols(df, target)
        if not feature_cols:
            return None

        high_card: list[str] = []
        for col in feature_cols:
            series = df[col]
            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                non_null = series.dropna()
                n_total = len(non_null)
                if n_total > 20:
                    try:
                        n_uniq = non_null.nunique()
                    except TypeError:
                        n_uniq = len(non_null)
                    ratio = n_uniq / n_total
                    if ratio > 0.50:
                        high_card.append(col)

        if high_card:
            return ValidationMessage(
                severity=self.default_severity,
                title="High Cardinality Feature Detected",
                description=f"Found high-cardinality categorical feature(s) (>50% unique): {high_card}",
                suggestion="Use target encoding, frequency encoding, or feature grouping.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"high_cardinality_features": high_card},
            )
        return None


class IdentifierFeatureRule(ValidationRule):
    rule_id = "KML-S006"
    name = "Identifier Feature Check"
    description = "Detect unique row IDs, UUIDs, or primary keys in feature columns."
    default_severity = ValidationSeverity.WARNING

    def check(self, df: pd.DataFrame, target: str | None = None, **kwargs: Any) -> ValidationMessage | None:
        feature_cols = _get_feature_cols(df, target)
        if not feature_cols:
            return None

        id_features: list[str] = []
        for col in feature_cols:
            series = df[col]
            non_null = series.dropna()
            n_total = len(non_null)
            if n_total > 10:
                ratio = non_null.nunique() / n_total
                col_lower = col.lower().replace(" ", "_")
                is_id_name = any(kw in col_lower for kw in _ID_KEYWORDS)
                if ratio > 0.95 and is_id_name:
                    id_features.append(col)

        if id_features:
            return ValidationMessage(
                severity=self.default_severity,
                title="Identifier Feature Detected",
                description=f"Found identifier feature(s) matching primary key patterns: {id_features}",
                suggestion="Exclude identifier columns from training features to prevent data leakage.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"identifier_features": id_features},
            )
        return None


class DatetimeFeatureRule(ValidationRule):
    rule_id = "KML-S007"
    name = "Datetime Feature Check"
    description = "Identify date/datetime formatted feature columns."
    default_severity = ValidationSeverity.INFO

    def check(self, df: pd.DataFrame, target: str | None = None, **kwargs: Any) -> ValidationMessage | None:
        feature_cols = _get_feature_cols(df, target)
        if not feature_cols:
            return None

        dt_features: list[str] = []
        for col in feature_cols:
            series = df[col]
            if pd.api.types.is_datetime64_any_dtype(series):
                dt_features.append(col)

        if dt_features:
            return ValidationMessage(
                severity=self.default_severity,
                title="Datetime Feature Identified",
                description=f"Identified {len(dt_features)} datetime feature(s): {dt_features}",
                suggestion="Extract date components (Year, Month, Day, Weekday, Hour).",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"datetime_features": dt_features},
            )
        return None


class BooleanFeatureRule(ValidationRule):
    rule_id = "KML-S008"
    name = "Boolean Feature Check"
    description = "Identify binary boolean or true/false feature columns."
    default_severity = ValidationSeverity.INFO

    def check(self, df: pd.DataFrame, target: str | None = None, **kwargs: Any) -> ValidationMessage | None:
        feature_cols = _get_feature_cols(df, target)
        if not feature_cols:
            return None

        bool_features: list[str] = []
        for col in feature_cols:
            series = df[col]
            if pd.api.types.is_bool_dtype(series):
                bool_features.append(col)

        if bool_features:
            return ValidationMessage(
                severity=self.default_severity,
                title="Boolean Feature Identified",
                description=f"Identified {len(bool_features)} boolean feature(s): {bool_features}",
                suggestion="Encode as binary 0/1 features.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"boolean_features": bool_features},
            )
        return None


class TextFeatureRule(ValidationRule):
    rule_id = "KML-S009"
    name = "Text Feature Check"
    description = "Identify free-text natural language feature columns."
    default_severity = ValidationSeverity.INFO

    def check(self, df: pd.DataFrame, target: str | None = None, **kwargs: Any) -> ValidationMessage | None:
        feature_cols = _get_feature_cols(df, target)
        if not feature_cols:
            return None

        text_features: list[str] = []
        for col in feature_cols:
            series = df[col]
            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                sample = series.dropna().head(50).astype(str)
                if len(sample) > 0:
                    avg_words = float(sample.str.split().str.len().mean())
                    if avg_words > 5:
                        text_features.append(col)

        if text_features:
            return ValidationMessage(
                severity=self.default_severity,
                title="Text Feature Identified",
                description=f"Identified {len(text_features)} free-text feature(s): {text_features}",
                suggestion="Apply TF-IDF vectorization or text embeddings.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"text_features": text_features},
            )
        return None


class MixedDatatypeRule(ValidationRule):
    rule_id = "KML-S010"
    name = "Mixed Datatype Check"
    description = "Detect mixed Python types within an object feature column."
    default_severity = ValidationSeverity.WARNING

    def check(self, df: pd.DataFrame, target: str | None = None, **kwargs: Any) -> ValidationMessage | None:
        feature_cols = _get_feature_cols(df, target)
        if not feature_cols:
            return None

        mixed_features: list[str] = []
        for col in feature_cols:
            series = df[col]
            if pd.api.types.is_object_dtype(series):
                non_null = series.dropna()
                if len(non_null) > 0:
                    type_counts = non_null.map(type).nunique()
                    if type_counts > 1:
                        mixed_features.append(col)

        if mixed_features:
            return ValidationMessage(
                severity=self.default_severity,
                title="Mixed Datatypes in Feature Column",
                description=f"Found mixed types in feature column(s): {mixed_features}",
                suggestion="Cast feature column to a consistent type (e.g. str or float).",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"mixed_features": mixed_features},
            )
        return None


class InfiniteNumericFeatureRule(ValidationRule):
    rule_id = "KML-S011"
    name = "Infinite Numeric Values Check"
    description = "Detect inf or -inf values in numeric feature columns."
    default_severity = ValidationSeverity.ERROR

    def check(self, df: pd.DataFrame, target: str | None = None, **kwargs: Any) -> ValidationMessage | None:
        feature_cols = _get_feature_cols(df, target)
        if not feature_cols:
            return None

        inf_cols: dict[str, int] = {}
        for col in feature_cols:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                try:
                    inf_count = int(np.isinf(series).sum())
                    if inf_count > 0:
                        inf_cols[col] = inf_count
                except Exception:
                    continue

        if inf_cols:
            return ValidationMessage(
                severity=self.default_severity,
                title="Infinite Values in Feature Column",
                description=f"Found infinite values in numeric features: {inf_cols}",
                suggestion="Replace infinite values with NaN or finite numerical bounds.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"infinite_features": inf_cols},
            )
        return None


class ExtremelySparseFeatureRule(ValidationRule):
    rule_id = "KML-S012"
    name = "Extremely Sparse Feature Check"
    description = "Detect feature columns with missing percentage > 70%."
    default_severity = ValidationSeverity.WARNING

    def check(self, df: pd.DataFrame, target: str | None = None, **kwargs: Any) -> ValidationMessage | None:
        feature_cols = _get_feature_cols(df, target)
        if not feature_cols:
            return None

        n_rows = len(df)
        if n_rows == 0:
            return None

        sparse_cols: dict[str, float] = {}
        for col in feature_cols:
            null_ratio = df[col].isna().sum() / n_rows
            if null_ratio > 0.70:
                sparse_cols[col] = round(null_ratio * 100, 1)

        if sparse_cols:
            return ValidationMessage(
                severity=self.default_severity,
                title="Extremely Sparse Feature (>70% Missing)",
                description=f"Found feature(s) with >70% missing values: {sparse_cols}",
                suggestion="Consider dropping extremely sparse features or using domain-specific imputation.",
                rule_id=self.rule_id,
                code=self.rule_id,
                context={"sparse_features": sparse_cols},
            )
        return None
