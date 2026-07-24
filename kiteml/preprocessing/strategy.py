"""
strategy.py — Preprocessing strategy enums for KiteML.
"""

from enum import Enum


class MissingStrategy(str, Enum):
    """Strategy for handling missing values."""

    MEDIAN = "median"
    MEAN = "mean"
    MODE = "mode"
    CONSTANT = "constant"
    DROP = "drop"
    NONE = "none"


class EncodingStrategy(str, Enum):
    """Strategy for categorical feature encoding."""

    ONE_HOT = "one_hot"
    TARGET = "target"
    ORDINAL = "ordinal"
    FREQUENCY = "frequency"
    NONE = "none"


class ScalingStrategy(str, Enum):
    """Strategy for numerical feature scaling."""

    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"
    NONE = "none"


class DatetimeStrategy(str, Enum):
    """Strategy for datetime feature extraction."""

    EXTRACT_COMPONENTS = "extract_components"
    TIMESTAMP = "timestamp"
    NONE = "none"


class TextStrategy(str, Enum):
    """Strategy for text feature vectorization."""

    TFIDF = "tfidf"
    COUNT = "count"
    NONE = "none"
