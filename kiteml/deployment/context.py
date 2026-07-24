"""
context.py — DeploymentContext shared state model for KiteML deployment subsystem.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DeploymentContext:
    """
    Shared state container holding packaging paths, manifests, and active adapters.
    """

    package_path: Path | None = None
    model_name: str = ""
    task_type: str = ""
    manifest: dict[str, Any] | None = None
    descriptor: dict[str, Any] | None = None
