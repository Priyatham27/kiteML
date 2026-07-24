"""
loader.py — load() root entry point for restoring .kiteml packages in KiteML.
"""

from pathlib import Path
from typing import Any

from kiteml.deployment.engine import DeploymentEngine
from kiteml.deployment.loader import LoadedPackage


def load(package_path: Path | str) -> LoadedPackage:
    """
    Load and restore full inference solution from .kiteml deployment package.

    Parameters
    ----------
    package_path : Path | str
        Path to .kiteml archive package file.

    Returns
    -------
    LoadedPackage
        Loaded package container ready for prediction.
    """
    engine = DeploymentEngine()
    return engine.load(package_path)
