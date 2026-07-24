"""
packager.py — Packager helper module for KiteML deployment subsystem.
"""

from pathlib import Path
from typing import Any

from kiteml.deployment.builder import PackageBuilder


class ModelPackager:
    """
    High-level packager delegating archive creation to PackageBuilder.
    """

    def __init__(self) -> None:
        self.builder = PackageBuilder()

    def package_model(
        self,
        model: Any,
        model_name: str,
        task_type: str,
        output_path: Path | str,
        pipeline: Any | None = None,
        feature_names: list[str] | None = None,
    ) -> Path:
        """Create package archive."""
        return self.builder.build_package(
            model=model,
            model_name=model_name,
            task_type=task_type,
            output_path=output_path,
            pipeline=pipeline,
            feature_names=feature_names,
        )
