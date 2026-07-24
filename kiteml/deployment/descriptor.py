"""
descriptor.py — UniversalDeploymentDescriptor Flagship Feature for Story 5.7.
"""

import platform
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class UniversalDeploymentDescriptor:
    """
    Separates trained ML artifacts from execution environments with explicit contracts and compatibility matrices.
    """

    model_name: str
    task_type: str
    runtime_python_version: str = platform.python_version()
    kiteml_version: str = "1.0.2"
    supported_adapters: list[str] = field(default_factory=lambda: ["fastapi", "joblib", "pickle"])
    feature_names: list[str] = field(default_factory=list)
    prediction_contract: dict[str, str] = field(
        default_factory=lambda: {"input_format": "json_or_dataframe", "output_format": "array_or_json"}
    )
    hardware_requirements: dict[str, str] = field(
        default_factory=lambda: {"min_ram_mb": "256", "cuda_required": "false"}
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize deployment descriptor to dictionary."""
        return asdict(self)
