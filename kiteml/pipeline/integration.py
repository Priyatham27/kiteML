"""
integration.py — Integration helpers for connecting DX Pipeline with KiteML subsystems.
"""

from typing import Any

from kiteml.pipeline.dx_pipeline import DXPipeline


def create_dx_pipeline() -> DXPipeline:
    """Factory creating a new DXPipeline instance."""
    pipeline = DXPipeline()
    pipeline.start()
    return pipeline
