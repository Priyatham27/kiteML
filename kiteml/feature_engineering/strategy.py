"""
strategy.py — Feature engineering transformation type enums for KiteML.
"""

from enum import Enum


class FETransformType(str, Enum):
    """Types of feature engineering transformations."""

    DATETIME_YEAR = "datetime_year"
    DATETIME_MONTH = "datetime_month"
    DATETIME_DAY = "datetime_day"
    DATETIME_WEEKDAY = "datetime_weekday"
    DATETIME_QUARTER = "datetime_quarter"
    DATETIME_IS_WEEKEND = "datetime_is_weekend"

    LOG_TRANSFORM = "log_transform"
    SQRT_TRANSFORM = "sqrt_transform"
    SQUARE_TRANSFORM = "square_transform"

    INTERACTION_PRODUCT = "interaction_product"
    INTERACTION_RATIO = "interaction_ratio"

    CATEGORY_FREQUENCY = "category_frequency"

    TEXT_WORD_COUNT = "text_word_count"
    TEXT_CHAR_COUNT = "text_char_count"
    TEXT_AVG_WORD_LEN = "text_avg_word_len"
