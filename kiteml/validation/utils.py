"""
utils.py — Helper utilities for the KiteML validation framework.
"""

import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import pandas as pd


def get_dataframe_memory_mb(df: pd.DataFrame) -> float:
    """Calculate the total memory usage of a DataFrame in Megabytes (MB)."""
    if not isinstance(df, pd.DataFrame):
        return 0.0
    return float(df.memory_usage(deep=True).sum()) / (1024 * 1024)


def format_bytes(num_bytes: float) -> str:
    """Format byte count into human-readable string (KB, MB, GB)."""
    if num_bytes < 1024:
        return f"{num_bytes:.0f} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    elif num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"


def format_number(val: int | float) -> str:
    """Format large numbers with thousand separators."""
    if isinstance(val, int):
        return f"{val:,}"
    return f"{val:,.2f}"


@contextmanager
def timer() -> Generator[dict[str, float], None, None]:
    """
    Context manager to measure execution time in seconds.

    Yields
    ------
    dict[str, float]
        Dictionary with key 'elapsed' updated when the block completes.
    """
    result: dict[str, float] = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = round(time.perf_counter() - start, 6)
