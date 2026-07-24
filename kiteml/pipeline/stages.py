"""
stages.py — PipelineStage base class and concrete transformation stages for KiteML.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from kiteml.pipeline.context import PipelineContext


class PipelineStage(ABC):
    """Abstract base class for all transformation pipeline stages."""

    name: str = "PipelineStage"
    priority: int = 50

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Any = None, context: PipelineContext | None = None) -> "PipelineStage":
        """Fit stage transformers on training data."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame, context: PipelineContext | None = None) -> pd.DataFrame:
        """Transform input dataset using fitted transformers."""
        pass

    def fit_transform(self, X: pd.DataFrame, y: Any = None, context: PipelineContext | None = None) -> pd.DataFrame:
        """Fit transformers and transform input dataset."""
        return self.fit(X, y=y, context=context).transform(X, context=context)

    def validate(self, X: pd.DataFrame) -> bool:
        """Validate input DataFrame readiness for stage transformation."""
        return isinstance(X, pd.DataFrame)


class MissingValueStage(PipelineStage):
    """Stage executing missing value imputation."""

    name = "MissingValueStage"
    priority = 10

    def __init__(self) -> None:
        self.num_imputer: SimpleImputer | None = None
        self.cat_imputer: SimpleImputer | None = None
        self.num_cols: list[str] = []
        self.cat_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None, context: PipelineContext | None = None) -> "MissingValueStage":
        self.num_cols = X.select_dtypes(include=["number"]).columns.tolist()
        self.cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

        if self.num_cols:
            self.num_imputer = SimpleImputer(strategy="median")
            self.num_imputer.fit(X[self.num_cols])

        if self.cat_cols:
            self.cat_imputer = SimpleImputer(strategy="most_frequent")
            self.cat_imputer.fit(X[self.cat_cols])

        return self

    def transform(self, X: pd.DataFrame, context: PipelineContext | None = None) -> pd.DataFrame:
        X_out = X.copy()

        if self.num_imputer and self.num_cols:
            valid_num = [c for c in self.num_cols if c in X_out.columns]
            if valid_num:
                imputed = self.num_imputer.transform(X_out[valid_num])
                X_out[valid_num] = imputed

        if self.cat_imputer and self.cat_cols:
            valid_cat = [c for c in self.cat_cols if c in X_out.columns]
            if valid_cat:
                imputed = self.cat_imputer.transform(X_out[valid_cat])
                X_out[valid_cat] = imputed

        return X_out


class DatetimeStage(PipelineStage):
    """Stage executing datetime component extractions."""

    name = "DatetimeStage"
    priority = 20

    def __init__(self) -> None:
        self.datetime_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None, context: PipelineContext | None = None) -> "DatetimeStage":
        self.datetime_cols = X.select_dtypes(include=["datetime", "datetime64"]).columns.tolist()
        return self

    def transform(self, X: pd.DataFrame, context: PipelineContext | None = None) -> pd.DataFrame:
        X_out = X.copy()

        for col in self.datetime_cols:
            if col in X_out.columns and pd.api.types.is_datetime64_any_dtype(X_out[col]):
                dt_series = X_out[col].dt
                X_out[f"{col}_year"] = dt_series.year
                X_out[f"{col}_month"] = dt_series.month
                X_out[f"{col}_day"] = dt_series.day
                X_out[f"{col}_weekday"] = dt_series.weekday
                X_out[f"{col}_quarter"] = dt_series.quarter
                X_out[f"{col}_is_weekend"] = (dt_series.weekday >= 5).astype(int)
                X_out = X_out.drop(columns=[col])

        return X_out


class EncodingStage(PipelineStage):
    """Stage executing categorical encoding."""

    name = "EncodingStage"
    priority = 30

    def __init__(self) -> None:
        self.ohe: OneHotEncoder | None = None
        self.cat_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None, context: PipelineContext | None = None) -> "EncodingStage":
        self.cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

        if self.cat_cols:
            self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            self.ohe.fit(X[self.cat_cols].astype(str))

        return self

    def transform(self, X: pd.DataFrame, context: PipelineContext | None = None) -> pd.DataFrame:
        X_out = X.copy()

        if self.ohe and self.cat_cols:
            valid_cat = [c for c in self.cat_cols if c in X_out.columns]
            if valid_cat:
                encoded = self.ohe.transform(X_out[valid_cat].astype(str))
                encoded_cols = self.ohe.get_feature_names_out(valid_cat)
                encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=X_out.index)
                X_out = X_out.drop(columns=valid_cat)
                X_out = pd.concat([X_out, encoded_df], axis=1)

        return X_out


class ScalingStage(PipelineStage):
    """Stage executing numerical feature scaling."""

    name = "ScalingStage"
    priority = 50

    def __init__(self) -> None:
        self.scaler: StandardScaler | None = None
        self.num_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None, context: PipelineContext | None = None) -> "ScalingStage":
        self.num_cols = X.select_dtypes(include=["number"]).columns.tolist()

        if self.num_cols:
            self.scaler = StandardScaler()
            self.scaler.fit(X[self.num_cols])

        return self

    def transform(self, X: pd.DataFrame, context: PipelineContext | None = None) -> pd.DataFrame:
        X_out = X.copy()

        if self.scaler and self.num_cols:
            valid_num = [c for c in self.num_cols if c in X_out.columns]
            if valid_num:
                scaled = self.scaler.transform(X_out[valid_num])
                X_out[valid_num] = scaled

        return X_out


class FeatureEngineeringStage(PipelineStage):
    """Stage executing planned feature engineering candidates."""

    name = "FeatureEngineeringStage"
    priority = 60

    def fit(self, X: pd.DataFrame, y: Any = None, context: PipelineContext | None = None) -> "FeatureEngineeringStage":
        return self

    def transform(self, X: pd.DataFrame, context: PipelineContext | None = None) -> pd.DataFrame:
        X_out = X.copy()

        if context and context.engineering_blueprint:
            fe_plans = getattr(context.engineering_blueprint, "feature_plans", {})
            for plan in fe_plans.values():
                if not getattr(plan, "enabled", True):
                    continue

                sources = getattr(plan, "source_columns", [])
                tt = str(getattr(plan, "transform_type", ""))
                gen_name = getattr(plan, "generated_name", "")

                if tt == "interaction_product" and len(sources) == 2:
                    s1, s2 = sources[0], sources[1]
                    if (
                        s1 in X_out.columns
                        and s2 in X_out.columns
                        and pd.api.types.is_numeric_dtype(X_out[s1])
                        and pd.api.types.is_numeric_dtype(X_out[s2])
                    ):
                        X_out[gen_name] = X_out[s1] * X_out[s2]

                elif tt == "log_transform" and len(sources) == 1:
                    s1 = sources[0]
                    if s1 in X_out.columns and pd.api.types.is_numeric_dtype(X_out[s1]):
                        X_out[gen_name] = np.log1p(np.maximum(0, X_out[s1]))

        return X_out


class FeatureSelectionStage(PipelineStage):
    """Stage executing feature selection column filtering."""

    name = "FeatureSelectionStage"
    priority = 70

    def fit(self, X: pd.DataFrame, y: Any = None, context: PipelineContext | None = None) -> "FeatureSelectionStage":
        return self

    def transform(self, X: pd.DataFrame, context: PipelineContext | None = None) -> pd.DataFrame:
        X_out = X.copy()

        if context and context.selection_blueprint:
            removed = getattr(context.selection_blueprint, "removed_features", [])
            cols_to_drop = [c for c in removed if c in X_out.columns]
            if cols_to_drop:
                X_out = X_out.drop(columns=cols_to_drop)

        return X_out
