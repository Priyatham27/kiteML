"""
model_info.py — ModelInfo metadata data model for KiteML model registry.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelInfo:
    """
    Metadata describing algorithm characteristics, task support, and capabilities.
    """

    name: str
    family: str
    task_types: list[str]
    supports_probability: bool = False
    supports_missing_values: bool = False
    supports_categorical: bool = False
    supports_sparse: bool = False
    supports_multiclass: bool = True
    deterministic: bool = True
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    default_params: dict[str, Any] = field(default_factory=dict)

    def supports_task(self, task_type: str) -> bool:
        """Check if model supports specified task type."""
        return task_type in self.task_types or any(t in task_type for t in self.task_types)
