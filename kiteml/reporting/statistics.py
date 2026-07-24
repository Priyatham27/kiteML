"""
statistics.py — TransformationStatistics data model for KiteML pipeline reporting.
"""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TransformationStatistics:
    """
    Measurable dataset transformation statistics and performance metrics.
    """

    initial_rows: int = 0
    final_rows: int = 0
    initial_cols: int = 0
    final_cols: int = 0
    generated_features_count: int = 0
    dropped_features_count: int = 0
    encoded_features_count: int = 0
    imputed_values_count: int = 0
    total_execution_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize statistics to dictionary."""
        return asdict(self)
