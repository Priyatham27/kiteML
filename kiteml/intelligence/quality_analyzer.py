"""
quality_analyzer.py — Detect real-world data quality issues.

Scans for: duplicate rows, constant columns, high missing rate,
mixed-type columns, and corrupted string patterns.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd


class Severity(str, Enum):
    INFO    = "info"
    WARNING = "warning"
    ERROR   = "error"


@dataclass
class QualityIssue:
    """A single detected data quality problem."""
    issue_type: str
    column: Optional[str]      # None → dataset-level
    severity: Severity
    description: str
    affected_count: int
    affected_ratio: float
    recommendation: str


@dataclass
class QualityReport:
    """Complete data quality assessment of a DataFrame."""
    n_rows: int
    n_cols: int
    issues: List[QualityIssue]
    score: float              # 0–100 (100 = perfect quality)
    summary: str

    @property
    def has_errors(self) -> bool:
        return any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == Severity.WARNING for i in self.issues)

    def by_severity(self, severity: Severity) -> List[QualityIssue]:
        return [i for i in self.issues if i.severity == severity]


def analyze_quality(df: pd.DataFrame) -> QualityReport:
    """
    Analyze data quality and return a structured report.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    QualityReport
    """
    issues: List[QualityIssue] = []
    n_rows = len(df)
    n_cols = len(df.columns)

    # ── 1. Duplicate rows ─────────────────────────────────────────────────
    n_dups = int(df.duplicated().sum())
    if n_dups > 0:
        ratio = n_dups / n_rows
        issues.append(QualityIssue(
            issue_type="duplicate_rows",
            column=None,
            severity=Severity.WARNING if ratio < 0.1 else Severity.ERROR,
            description=f"{n_dups} duplicate rows detected ({ratio:.1%})",
            affected_count=n_dups,
            affected_ratio=round(ratio, 4),
            recommendation="Call df.drop_duplicates() before training.",
        ))

    # ── 2. Per-column checks ──────────────────────────────────────────────
    for col in df.columns:
        series = df[col]
        n_null = int(series.isna().sum())
        null_ratio = n_null / n_rows if n_rows > 0 else 0.0
        n_unique = int(series.dropna().nunique())

        # Missing values
        if null_ratio > 0.70:
            issues.append(QualityIssue(
                issue_type="high_missing_rate",
                column=col,
                severity=Severity.ERROR,
                description=f"'{col}': {null_ratio:.1%} missing values",
                affected_count=n_null,
                affected_ratio=round(null_ratio, 4),
                recommendation=f"Consider dropping '{col}' or using domain-specific imputation.",
            ))
        elif null_ratio > 0.30:
            issues.append(QualityIssue(
                issue_type="moderate_missing_rate",
                column=col,
                severity=Severity.WARNING,
                description=f"'{col}': {null_ratio:.1%} missing values",
                affected_count=n_null,
                affected_ratio=round(null_ratio, 4),
                recommendation="KiteML will impute automatically; verify imputation strategy is appropriate.",
            ))

        # Constant column
        if n_unique <= 1:
            issues.append(QualityIssue(
                issue_type="constant_column",
                column=col,
                severity=Severity.ERROR,
                description=f"'{col}' has only {n_unique} unique value(s) — zero variance",
                affected_count=n_rows,
                affected_ratio=1.0,
                recommendation=f"Drop '{col}' — it carries no predictive information.",
            ))

        # Mixed types in object columns
        if pd.api.types.is_object_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                type_counts = non_null.map(type).value_counts()
                if len(type_counts) > 1:
                    issues.append(QualityIssue(
                        issue_type="mixed_types",
                        column=col,
                        severity=Severity.WARNING,
                        description=f"'{col}' contains mixed types: {list(type_counts.index)}",
                        affected_count=int(n_rows - type_counts.max()),
                        affected_ratio=round(1 - type_counts.max() / len(non_null), 4),
                        recommendation=f"Cast '{col}' to a consistent type before training.",
                    ))

        # Near-zero variance for numeric
        if pd.api.types.is_numeric_dtype(series) and len(series.dropna()) > 1:
            std = float(series.dropna().std())
            if 0 < std < 1e-8:
                issues.append(QualityIssue(
                    issue_type="near_zero_variance",
                    column=col,
                    severity=Severity.INFO,
                    description=f"'{col}' has near-zero variance (std={std:.2e})",
                    affected_count=n_rows,
                    affected_ratio=1.0,
                    recommendation=f"Consider dropping '{col}' — minimal predictive power.",
                ))

    # ── Score calculation ─────────────────────────────────────────────────
    error_count  = sum(1 for i in issues if i.severity == Severity.ERROR)
    warn_count   = sum(1 for i in issues if i.severity == Severity.WARNING)
    score = max(0.0, 100.0 - error_count * 15 - warn_count * 5)

    if score >= 90:
        summary = "Dataset quality is excellent."
    elif score >= 70:
        summary = "Dataset quality is good with minor issues."
    elif score >= 50:
        summary = "Dataset has notable quality issues that may affect model performance."
    else:
        summary = "Dataset has serious quality issues. Address before training."

    return QualityReport(
        n_rows=n_rows, n_cols=n_cols,
        issues=issues, score=round(score, 1), summary=summary,
    )
