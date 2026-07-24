"""
context.py — OrchestrationContext shared pipeline state model for KiteML.
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class OrchestrationContext:
    """
    Shared context passing pipeline state across orchestration stages.
    """

    dataset: pd.DataFrame | None = None
    target_name: str | None = None
    problem_type: str | None = None
    keep_features: list[str] = field(default_factory=list)
    data_profile: Any | None = None
    preprocessing_blueprint: Any | None = None
    engineering_blueprint: Any | None = None
    selection_blueprint: Any | None = None
    transformed_df: pd.DataFrame | None = None
    pipeline: Any | None = None
    report: Any | None = None
    diagnostics: Any | None = None
