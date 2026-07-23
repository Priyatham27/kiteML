"""
feature_profile.py — Structured feature profile metadata representation for KiteML.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FeatureProfile:
    """
    Structured profile and metadata representation for a single feature column.

    Attributes
    ----------
    name : str
        Name of the feature column.
    dtype : str
        Pandas data type of the column.
    semantic_type : str
        Inferred semantic type (numeric, categorical, boolean, datetime, text, identifier, constant).
    missing_count : int
        Number of missing (NaN) values in the feature.
    missing_percentage : float
        Percentage of missing values (0–100%).
    unique_count : int
        Number of unique non-null values.
    unique_ratio : float
        Ratio of unique values to non-null count (0–1.0).
    cardinality : str
        Cardinality tier ('low', 'medium', 'high', 'identifier').
    is_constant : bool
        True if feature has <= 1 unique value.
    is_identifier : bool
        True if feature appears to be a unique row ID or UUID.
    is_datetime : bool
        True if feature is date/time formatted.
    is_text : bool
        True if feature is free-form text.
    is_boolean : bool
        True if feature contains binary True/False, 0/1, or Yes/No values.
    recommendation : str
        Suggested processing recommendation for pipeline/intelligence.
    health_score : int
        Feature health score (0–100).
    notes : list[str]
        Detailed diagnostic notes.
    """

    name: str
    dtype: str
    semantic_type: str = "unknown"
    missing_count: int = 0
    missing_percentage: float = 0.0
    unique_count: int = 0
    unique_ratio: float = 0.0
    cardinality: str = "low"
    is_constant: bool = False
    is_identifier: bool = False
    is_datetime: bool = False
    is_text: bool = False
    is_boolean: bool = False
    recommendation: str = "Keep"
    health_score: int = 100
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert feature profile into a dictionary."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "semantic_type": self.semantic_type,
            "missing_count": self.missing_count,
            "missing_percentage": self.missing_percentage,
            "unique_count": self.unique_count,
            "unique_ratio": self.unique_ratio,
            "cardinality": self.cardinality,
            "is_constant": self.is_constant,
            "is_identifier": self.is_identifier,
            "is_datetime": self.is_datetime,
            "is_text": self.is_text,
            "is_boolean": self.is_boolean,
            "recommendation": self.recommendation,
            "health_score": self.health_score,
            "notes": self.notes,
        }
