"""
utils.py — Helper functions for the KiteML warning framework.
"""

import warnings
from typing import Any

from kiteml.warnings.base import KiteMLWarning


def emit_warning(warning: KiteMLWarning, stacklevel: int = 2) -> None:
    """Emit a KiteMLWarning through Python's standard warnings module."""
    warnings.warn(warning, stacklevel=stacklevel)
