"""
rules.py — Centralized FERuleEngine and feature engineering thresholds for KiteML.
"""

from dataclasses import dataclass


@dataclass
class FERuleEngine:
    """
    Centralized rule engine holding decision thresholds for feature engineering planning.
    """

    skewness_threshold: float = 1.5
    max_interaction_pairs: int = 10
    min_text_avg_words: float = 3.0
    enable_datetime_extractions: bool = True
    enable_numeric_transforms: bool = True
    enable_interactions: bool = True
    enable_text_derived: bool = True


default_fe_rule_engine = FERuleEngine()
