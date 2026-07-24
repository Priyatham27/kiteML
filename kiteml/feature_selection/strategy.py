"""
strategy.py — Feature selection decision enums for KiteML.
"""

from enum import Enum


class SelectionDecision(str, Enum):
    """Decision outcomes for feature selection."""

    KEEP = "keep"
    REMOVE = "remove"
    FLAG = "flag"
    DEFER = "defer"
