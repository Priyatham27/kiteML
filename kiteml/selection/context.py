"""
context.py — SelectionContext shared state model for KiteML model selection.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SelectionContext:
    """
    Shared selection state holding evaluated candidate records and policy settings.
    """

    candidates: list[dict[str, Any]] = field(default_factory=list)
    policy: str = "balanced"
    ranked_candidates: list[dict[str, Any]] = field(default_factory=list)
    pareto_frontier: dict[str, Any] = field(default_factory=dict)
