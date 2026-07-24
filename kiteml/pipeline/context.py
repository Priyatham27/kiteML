"""
context.py — PipelineContext shared state model for KiteML transformation pipeline.
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class PipelineContext:
    """
    Shared state container passed across pipeline execution stages.
    """

    original_df: pd.DataFrame | None = None
    current_df: pd.DataFrame | None = None
    target_name: str | None = None
    problem_type: str | None = None
    preprocessing_blueprint: Any | None = None
    engineering_blueprint: Any | None = None
    selection_blueprint: Any | None = None
    fitted_transformers: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
