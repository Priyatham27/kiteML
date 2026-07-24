"""
context.py — MLContext global shared state model for KiteML Unified ML Engine.
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class MLContext:
    """
    Global shared state context tracking artifacts across the end-to-end ML training DAG.
    """

    dataframe: pd.DataFrame
    target_column: str
    problem_type: str = "classification"

    # Stage outputs
    profile: Any = None
    validation_report: Any = None
    pipeline_result: Any = None
    trained_models: list[dict[str, Any]] = field(default_factory=list)
    optimized_models: list[dict[str, Any]] = field(default_factory=list)
    evaluation_reports: list[dict[str, Any]] = field(default_factory=list)
    best_model: Any = None
    deployment_package: Any = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
