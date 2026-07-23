"""
providers.py — Specialized SuggestionProvider classes for KiteML suggestions engine.
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from kiteml.suggestions.context import SuggestionContext
from kiteml.suggestions.matcher import match_column_name
from kiteml.suggestions.result import Suggestion


class BaseSuggestionProvider(ABC):
    """Abstract base class for suggestion providers."""

    name: str = "BaseProvider"

    @abstractmethod
    def generate(self, context: SuggestionContext) -> list[Suggestion]:
        """Generate suggestions based on execution context."""
        pass


class ColumnSuggestionProvider(BaseSuggestionProvider):
    """Provider for column name typo matching and fuzzy column resolution."""

    name = "ColumnSuggestionProvider"

    def generate(self, context: SuggestionContext) -> list[Suggestion]:
        suggestions: list[Suggestion] = []
        target = context.target or (
            context.error.context.get("target") if context.error and hasattr(context.error, "context") else None
        )
        cols = context.available_columns

        if not target or not cols or target in cols:
            return suggestions

        matches = match_column_name(target, cols, threshold=0.4)
        for col_name, sim in matches[:3]:
            # Boost confidence for direct typo match (e.g. 'prcie' -> 'Price')
            conf = min(0.98, max(0.85, sim + 0.3)) if sim >= 0.5 else sim
            why_reasons = [
                f"Column name '{col_name}' closely matches your input '{target}'.",
                f"Similarity score is {int(sim * 100)}%.",
                f"'{col_name}' exists in the current dataset.",
            ]
            suggestions.append(
                Suggestion(
                    title=f"Did you mean '{col_name}'?",
                    description=f"Use existing column '{col_name}' as target feature.",
                    confidence=conf,
                    category="Column",
                    source=self.name,
                    action=f"target='{col_name}'",
                    why=why_reasons,
                )
            )
        return suggestions


class TargetSuggestionProvider(BaseSuggestionProvider):
    """Provider recommending optimal target columns based on dataset properties."""

    name = "TargetSuggestionProvider"

    def generate(self, context: SuggestionContext) -> list[Suggestion]:
        suggestions: list[Suggestion] = []
        df = context.df
        if not isinstance(df, pd.DataFrame) or df.empty or context.target in df.columns:
            return suggestions

        # Find potential target columns
        for col in df.columns:
            nunique = df[col].nunique(dropna=True)
            dtype = df[col].dtype
            col_lower = col.lower()

            is_named_target = col_lower in ("target", "label", "class", "y", "outcome")

            if nunique >= 2 and nunique <= 20:
                conf = 0.95 if is_named_target else (0.90 if nunique == 2 else 0.85)
                why_reasons = [
                    f"Column '{col}' has {nunique} unique class labels.",
                    "Ideal distribution for classification tasks.",
                    (
                        "No missing target values detected."
                        if df[col].isna().sum() == 0
                        else f"Low missing values count ({df[col].isna().sum()})."
                    ),
                ]
                suggestions.append(
                    Suggestion(
                        title=f"Use '{col}' for Classification",
                        description=f"Column '{col}' has {nunique} discrete classes suitable for classification.",
                        confidence=conf,
                        category="Target",
                        source=self.name,
                        action=f"target='{col}', problem_type='classification'",
                        why=why_reasons,
                    )
                )
            elif pd.api.types.is_numeric_dtype(dtype) and nunique > 20:
                conf = 0.95 if is_named_target else 0.80
                why_reasons = [
                    f"Column '{col}' is continuous numeric (type {dtype}).",
                    f"Contains {nunique} distinct values.",
                    "High variance indicates strong regression target potential.",
                ]
                suggestions.append(
                    Suggestion(
                        title=f"Use '{col}' for Regression",
                        description=f"Numeric column '{col}' with continuous values suitable for regression.",
                        confidence=conf,
                        category="Target",
                        source=self.name,
                        action=f"target='{col}', problem_type='regression'",
                        why=why_reasons,
                    )
                )

        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        return suggestions[:3]


class SchemaSuggestionProvider(BaseSuggestionProvider):
    """Provider suggesting schema optimizations (constant features, high cardinality)."""

    name = "SchemaSuggestionProvider"

    def generate(self, context: SuggestionContext) -> list[Suggestion]:
        suggestions: list[Suggestion] = []
        df = context.df
        if not isinstance(df, pd.DataFrame) or df.empty:
            return suggestions

        for col in df.columns:
            if col == context.target:
                continue
            nunique = df[col].nunique(dropna=True)
            if nunique == 1:
                why_reasons = [
                    f"Column '{col}' has 0 variance (constant value across all rows).",
                    "Constant features provide no predictive signal to machine learning models.",
                    "Removing constant features improves training speed and stability.",
                ]
                suggestions.append(
                    Suggestion(
                        title=f"Remove constant feature '{col}'",
                        description=f"Feature '{col}' contains only 1 unique value.",
                        confidence=0.95,
                        category="Schema",
                        source=self.name,
                        action=f"df = df.drop(columns=['{col}'])",
                        why=why_reasons,
                    )
                )

        return suggestions


class ValidationSuggestionProvider(BaseSuggestionProvider):
    """Provider recommending missing value imputation and outlier handling strategies."""

    name = "ValidationSuggestionProvider"

    def generate(self, context: SuggestionContext) -> list[Suggestion]:
        suggestions: list[Suggestion] = []
        df = context.df
        if not isinstance(df, pd.DataFrame) or df.empty:
            return suggestions

        for col in df.columns:
            missing = df[col].isna().sum()
            ratio = missing / len(df)
            if 0 < ratio <= 0.4:
                is_num = pd.api.types.is_numeric_dtype(df[col].dtype)
                strategy = "Median Imputation" if is_num else "Mode Imputation"
                why_reasons = [
                    f"Column '{col}' has {missing} missing values ({ratio:.1%}).",
                    f"{'Numeric' if is_num else 'Categorical'} datatype detected.",
                    f"{strategy} preserves distribution without dropping data rows.",
                ]
                suggestions.append(
                    Suggestion(
                        title=f"Apply {strategy} to '{col}'",
                        description=f"Impute {ratio:.1%} missing values in feature '{col}'.",
                        confidence=0.90,
                        category="Validation",
                        source=self.name,
                        action=f"Impute '{col}' using {'median' if is_num else 'mode'}",
                        why=why_reasons,
                    )
                )

        return suggestions[:3]


class TrainingSuggestionProvider(BaseSuggestionProvider):
    """Provider recommending optimal model types based on dataset dimensions."""

    name = "TrainingSuggestionProvider"

    def generate(self, context: SuggestionContext) -> list[Suggestion]:
        suggestions: list[Suggestion] = []
        df = context.df
        if not isinstance(df, pd.DataFrame) or df.empty:
            return suggestions

        rows = len(df)
        if rows < 500:
            why_reasons = [
                f"Small dataset detected ({rows} rows).",
                "Tree-based ensembles (Random Forest, Extra Trees) perform exceptionally well on small tabular datasets.",
                "Prevents overfitting compared to overly complex deep learning architectures.",
            ]
            suggestions.append(
                Suggestion(
                    title="Recommend Random Forest / Extra Trees",
                    description="Small structured dataset (<500 rows) is ideal for tree ensembles.",
                    confidence=0.88,
                    category="Training",
                    source=self.name,
                    action="models=['RandomForest', 'ExtraTrees']",
                    why=why_reasons,
                )
            )

        return suggestions


class DeploymentSuggestionProvider(BaseSuggestionProvider):
    """Provider recommending ONNX export and container packaging."""

    name = "DeploymentSuggestionProvider"

    def generate(self, context: SuggestionContext) -> list[Suggestion]:
        return [
            Suggestion(
                title="Export Model to ONNX Format",
                description="Export fitted model pipeline to cross-platform ONNX runtime format.",
                confidence=0.80,
                category="Deployment",
                source=self.name,
                action="result.export_onnx('model.onnx')",
                why=[
                    "ONNX enables ultra-fast inference with up to 3x throughput improvement.",
                    "Allows execution in C++, Rust, Go, or JavaScript environments without Python.",
                ],
            )
        ]


class PerformanceSuggestionProvider(BaseSuggestionProvider):
    """Provider recommending performance optimizations for large datasets."""

    name = "PerformanceSuggestionProvider"

    def generate(self, context: SuggestionContext) -> list[Suggestion]:
        suggestions: list[Suggestion] = []
        df = context.df
        if isinstance(df, pd.DataFrame) and len(df) > 100_000:
            suggestions.append(
                Suggestion(
                    title="Enable Parallel Multiprocessing",
                    description="Large dataset (>100k rows) detected.",
                    confidence=0.92,
                    category="Performance",
                    source=self.name,
                    action="n_jobs=-1",
                    why=[
                        f"Dataset has {len(df):,} rows.",
                        "Using all CPU cores accelerates cross-validation by up to 4x.",
                    ],
                )
            )
        return suggestions
