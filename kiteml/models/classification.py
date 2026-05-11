"""
classification.py — Classification model collection.

This module now delegates entirely to the central registry so there is
a single source of truth.  It is kept as a standalone file for backward
compatibility and for clarity when importing directly.
"""

from kiteml.models.registry import get_classification_models

__all__ = ["get_classification_models"]
