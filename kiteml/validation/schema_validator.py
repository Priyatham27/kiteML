"""
schema_validator.py — Composite Schema Validator for KiteML feature columns.
"""

from typing import Any

import pandas as pd

from kiteml.validation.feature_profile import FeatureProfile
from kiteml.validation.rules.schema_rules import (
    BooleanFeatureRule,
    ConstantFeatureRule,
    DatetimeFeatureRule,
    DuplicateFeatureNameRule,
    EmptyFeatureNameRule,
    ExtremelySparseFeatureRule,
    HighCardinalityFeatureRule,
    IdentifierFeatureRule,
    InfiniteNumericFeatureRule,
    MixedDatatypeRule,
    TextFeatureRule,
    UnsupportedDatatypeRule,
)
from kiteml.validation.validation_result import ValidationResult
from kiteml.validation.validator import BaseValidator

_ID_KEYWORDS = {"id", "uuid", "guid", "key", "index", "code", "ref", "no", "num", "number", "sku"}


class SchemaValidator(BaseValidator):
    """
    Validates feature column structure, datatypes, cardinality, semantic types,
    and generates FeatureProfile instances and feature processing recommendations.
    """

    description: str = "Validates feature column schema, datatypes, and generates recommendations."

    @property
    def name(self) -> str:
        return "SchemaValidator"

    def __init__(self, rules: list[Any] | None = None) -> None:
        if rules is None:
            rules = [
                EmptyFeatureNameRule(),
                DuplicateFeatureNameRule(),
                UnsupportedDatatypeRule(),
                ConstantFeatureRule(),
                HighCardinalityFeatureRule(),
                IdentifierFeatureRule(),
                DatetimeFeatureRule(),
                BooleanFeatureRule(),
                TextFeatureRule(),
                MixedDatatypeRule(),
                InfiniteNumericFeatureRule(),
                ExtremelySparseFeatureRule(),
            ]
        super().__init__(rules=rules)

    def validate(
        self,
        df: Any,
        target: str | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """
        Execute schema validation rules and generate feature profiles & recommendations.

        Parameters
        ----------
        df : Any
            Dataset to validate.
        target : str, optional
            Target column name (automatically excluded from feature validation).
        **kwargs : Any

        Returns
        -------
        ValidationResult
        """
        result = super().validate(df, target=target, **kwargs)

        # Build Feature Profiles & Recommendations if df is a valid DataFrame
        if isinstance(df, pd.DataFrame) and len(df) > 0 and len(df.columns) > 0:
            profiles: dict[str, FeatureProfile] = {}
            summary: dict[str, int] = {
                "numeric": 0,
                "categorical": 0,
                "boolean": 0,
                "datetime": 0,
                "text": 0,
                "identifier": 0,
                "constant": 0,
                "unknown": 0,
            }

            feature_cols = [c for c in df.columns if target is None or str(c) != str(target)]
            for col in feature_cols:
                prof = self._build_feature_profile(df[str(col)], str(col))
                profiles[str(col)] = prof
                summary[prof.semantic_type] = summary.get(prof.semantic_type, 0) + 1

            result.statistics["feature_profiles"] = {k: v.to_dict() for k, v in profiles.items()}
            result.statistics["schema_summary"] = summary
            result.statistics["n_features"] = len(feature_cols)

        # Compute Schema Health Score
        score, rating = self._compute_schema_health_score(result)
        result.statistics["health_score"] = score
        result.statistics["health_rating"] = rating

        return result

    def _build_feature_profile(self, series_input: Any, name: str) -> FeatureProfile:
        """Build detailed FeatureProfile for a single feature column."""
        series = series_input.iloc[:, 0] if isinstance(series_input, pd.DataFrame) else series_input

        n_total = len(series)
        n_null = int(series.isna().sum()) if n_total > 0 else 0
        missing_pct = round((n_null / n_total * 100), 2) if n_total > 0 else 0.0

        non_null = series.dropna()
        if len(non_null) > 0:
            try:
                n_unique = int(non_null.nunique())
            except TypeError:
                n_unique = len(non_null)
        else:
            n_unique = 0

        unique_ratio = round(n_unique / len(non_null), 4) if len(non_null) > 0 else 0.0

        dtype_str = str(series.dtype)
        name_lower = name.lower().replace(" ", "_")

        # Flag flags
        is_constant = n_unique <= 1
        is_id_name = any(kw in name_lower for kw in _ID_KEYWORDS)
        is_identifier = (unique_ratio > 0.95 and is_id_name) or (n_total > 50 and unique_ratio == 1.0 and is_id_name)
        is_datetime = pd.api.types.is_datetime64_any_dtype(series)
        is_bool = pd.api.types.is_bool_dtype(series)

        is_text = False
        if (
            not is_datetime
            and not is_bool
            and (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series))
        ):
            try:
                sample = non_null.head(50).astype(str)
                if len(sample) > 0:
                    avg_words = float(sample.str.split().str.len().mean())
                    if avg_words > 5:
                        is_text = True
            except Exception:
                pass

        # Determine Cardinality Tier
        if is_constant:
            cardinality = "constant"
        elif is_identifier:
            cardinality = "identifier"
        elif n_unique <= 10:
            cardinality = "low"
        elif unique_ratio > 0.50:
            cardinality = "high"
        elif unique_ratio > 0.20:
            cardinality = "medium"
        else:
            cardinality = "low"

        # Determine Semantic Type & Recommendation
        if is_constant:
            sem_type = "constant"
            rec = "Remove (Constant)"
            health = 50
        elif is_identifier:
            sem_type = "identifier"
            rec = "Remove (Identifier)"
            health = 70
        elif is_datetime:
            sem_type = "datetime"
            rec = "Extract Year/Month/Day"
            health = 100
        elif is_bool:
            sem_type = "boolean"
            rec = "Encode Binary (0/1)"
            health = 100
        elif is_text:
            sem_type = "text"
            rec = "TF-IDF Vectorization"
            health = 90
        elif pd.api.types.is_numeric_dtype(series):
            sem_type = "numeric"
            rec = "Standardize"
            health = 100
        elif (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            sem_type = "categorical"
            rec = "Target / Frequency Encode" if cardinality in ("high", "medium") else "One-Hot Encode"
            health = 95
        else:
            sem_type = "unknown"
            rec = "Verify Datatype"
            health = 60

        return FeatureProfile(
            name=name,
            dtype=dtype_str,
            semantic_type=sem_type,
            missing_count=n_null,
            missing_percentage=missing_pct,
            unique_count=n_unique,
            unique_ratio=unique_ratio,
            cardinality=cardinality,
            is_constant=is_constant,
            is_identifier=is_identifier,
            is_datetime=is_datetime,
            is_text=is_text,
            is_boolean=is_bool,
            recommendation=rec,
            health_score=health,
        )

    def _compute_schema_health_score(self, result: ValidationResult) -> tuple[int, str]:
        """Calculate Schema Health Score (0–100) and Rating."""
        if any(
            msg.rule_id in ("KML-S001", "KML-S002", "KML-S003", "KML-S011") and msg.severity == "error"
            for msg in result.messages
        ):
            return 40, "★☆☆☆☆ Poor"

        score = 100
        for msg in result.messages:
            rule_id = msg.rule_id
            if rule_id in ("KML-S001", "KML-S002", "KML-S003", "KML-S011"):
                score -= 15
            elif rule_id in ("KML-S004", "KML-S005", "KML-S006", "KML-S010", "KML-S012"):
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
