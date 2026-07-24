"""
rules.py — Centralized RuleEngine and preprocessing thresholds for KiteML.
"""

from dataclasses import dataclass


@dataclass
class RuleEngine:
    """
    Centralized rule engine holding decision thresholds for preprocessing planning.

    Attributes
    ----------
    low_cardinality_threshold : int
        Maximum unique values for One-Hot Encoding (default 15).
    high_missing_drop_threshold : float
        Missing ratio threshold above which a feature is ignored (default 0.80 / 80%).
    high_skewness_threshold : float
        Absolute skewness threshold above which RobustScaler is preferred over StandardScaler (default 1.5).
    """

    low_cardinality_threshold: int = 15
    high_missing_drop_threshold: float = 0.80
    high_skewness_threshold: float = 1.5


default_rule_engine = RuleEngine()
