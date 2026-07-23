"""
quality_profile.py — Structured quality profile metadata representation for KiteML.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class QualityProfile:
    """
    Structured dataset quality profile representation.

    Attributes
    ----------
    overall_score : int
        Quality health score from 0 to 100.
    overall_grade : str
        Letter grade ('A+', 'A', 'B', 'C', 'Needs Attention').
    health_rating : str
        Star rating display string ('★★★★★ Excellent', etc.).
    missing_summary : dict[str, Any]
        Summary of missing values and missing ratios.
    duplicate_summary : dict[str, Any]
        Summary of duplicate rows and ratios.
    outlier_summary : dict[str, Any]
        Summary of features containing statistical outliers.
    correlation_summary : dict[str, Any]
        Summary of highly correlated feature pairs.
    variance_summary : dict[str, Any]
        Summary of zero-variance or near-zero variance features.
    balance_summary : dict[str, Any]
        Summary of target class distribution and balance status.
    consistency_summary : dict[str, Any]
        Summary of string whitespace or pseudo-null consistency issues.
    memory_summary : dict[str, Any]
        Summary of dataset memory usage and column sizes.
    recommendations : list[str]
        List of actionable data quality recommendations.
    generated_at : str
        ISO format timestamp when profile was generated.
    """

    overall_score: int = 100
    overall_grade: str = "A+"
    health_rating: str = "★★★★★ Excellent"
    missing_summary: dict[str, Any] = field(default_factory=dict)
    duplicate_summary: dict[str, Any] = field(default_factory=dict)
    outlier_summary: dict[str, Any] = field(default_factory=dict)
    correlation_summary: dict[str, Any] = field(default_factory=dict)
    variance_summary: dict[str, Any] = field(default_factory=dict)
    balance_summary: dict[str, Any] = field(default_factory=dict)
    consistency_summary: dict[str, Any] = field(default_factory=dict)
    memory_summary: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert quality profile into a dictionary."""
        return {
            "overall_score": self.overall_score,
            "overall_grade": self.overall_grade,
            "health_rating": self.health_rating,
            "missing_summary": self.missing_summary,
            "duplicate_summary": self.duplicate_summary,
            "outlier_summary": self.outlier_summary,
            "correlation_summary": self.correlation_summary,
            "variance_summary": self.variance_summary,
            "balance_summary": self.balance_summary,
            "consistency_summary": self.consistency_summary,
            "memory_summary": self.memory_summary,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at,
        }
