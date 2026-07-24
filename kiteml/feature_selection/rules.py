"""
rules.py — Centralized FSRuleEngine and feature selection thresholds for KiteML.
"""

from dataclasses import dataclass, field


@dataclass
class FSRuleEngine:
    """
    Centralized rule engine holding thresholds and selector weights for feature selection.
    """

    max_correlation: float = 0.95
    min_variance: float = 1e-4
    max_missing_ratio: float = 0.80
    selector_weights: dict[str, float] = field(
        default_factory=lambda: {
            "RuleSelector": 0.25,
            "VarianceSelector": 0.15,
            "MissingValueSelector": 0.20,
            "CorrelationSelector": 0.20,
            "ImportanceEstimator": 0.20,
        }
    )


default_fs_rule_engine = FSRuleEngine()
