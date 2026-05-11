"""
telemetry/logging.py — Structured logging for KiteML CLI.
"""

import logging
import os
from typing import Optional


def setup_logger(name: str = "kiteml", log_file: Optional[str] = None, debug: bool = False) -> logging.Logger:
    """Configure structured logging for KiteML diagnostics."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Avoid duplicating handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
