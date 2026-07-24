"""
engine.py — DeploymentEngine master entry point for KiteML model packaging and deployment subsystem.
"""

from pathlib import Path
from typing import Any

from kiteml.deployment.adapters import DeploymentAdapter
from kiteml.deployment.builder import PackageBuilder
from kiteml.deployment.fastapi import FastAPIAdapter
from kiteml.deployment.joblib import JoblibAdapter
from kiteml.deployment.loader import LoadedPackage, PackageLoader
from kiteml.deployment.pickle import PickleAdapter
from kiteml.deployment.report import DeploymentReport
from kiteml.deployment.validator import PackageValidator


class DeploymentEngine:
    """
    Master DeploymentEngine executing model packaging, validation, loading, and framework exports.
    """

    def __init__(self) -> None:
        self.builder = PackageBuilder()
        self.validator = PackageValidator()
        self.loader = PackageLoader()
        self.adapters: dict[str, DeploymentAdapter] = {
            "fastapi": FastAPIAdapter(),
            "joblib": JoblibAdapter(),
            "pickle": PickleAdapter(),
        }

    def package(
        self,
        model: Any,
        model_name: str,
        task_type: str,
        output_path: Path | str,
        pipeline: Any | None = None,
        feature_names: list[str] | None = None,
    ) -> DeploymentReport:
        """
        Build and validate .kiteml package file.

        Parameters
        ----------
        model : Any
            Fitted model instance.
        model_name : str
            Model algorithm name.
        task_type : str
            ML task type.
        output_path : Path | str
            Destination file path.
        pipeline : Any, optional
            Fitted transformation pipeline instance.
        feature_names : list[str], optional
            Feature column names.

        Returns
        -------
        DeploymentReport
            Packaging summary report.
        """
        pkg_path = self.builder.build_package(
            model=model,
            model_name=model_name,
            task_type=task_type,
            output_path=output_path,
            pipeline=pipeline,
            feature_names=feature_names,
        )

        is_valid = self.validator.validate_package(pkg_path)
        size_kb = pkg_path.stat().st_size / 1024.0

        return DeploymentReport(
            package_path=str(pkg_path),
            is_valid=is_valid,
            model_name=model_name,
            task_type=task_type,
            file_size_kb=size_kb,
        )

    def load(self, package_path: Path | str) -> LoadedPackage:
        """
        Load and restore full inference environment from .kiteml archive.

        Parameters
        ----------
        package_path : Path | str
            Path to .kiteml package.

        Returns
        -------
        LoadedPackage
            Restored package object ready for prediction.
        """
        return self.loader.load_package(package_path)

    def export(
        self,
        package_path: Path | str,
        output_dir: Path | str,
        adapter: str = "fastapi",
    ) -> Path:
        """
        Export package to target deployment environment using adapter.

        Parameters
        ----------
        package_path : Path | str
            Path to .kiteml package.
        output_dir : Path | str
            Target directory for export.
        adapter : str
            Active deployment adapter name ('fastapi', 'joblib', 'pickle').

        Returns
        -------
        Path
            Path to exported files.
        """
        target_adapter = self.adapters.get(adapter.lower())
        if not target_adapter:
            raise ValueError(f"Unsupported deployment adapter '{adapter}'. Available: {list(self.adapters.keys())}")

        loaded = self.load(package_path)
        return target_adapter.export(
            model=loaded.model,
            output_dir=output_dir,
            pipeline=loaded.pipeline,
            metadata=loaded.manifest,
        )
