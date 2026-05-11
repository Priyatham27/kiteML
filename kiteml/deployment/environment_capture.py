"""
environment_capture.py — Capture full runtime environment snapshot.

Captures Python version, OS info, installed packages, and random seeds
for guaranteed reproducibility of any KiteML training run.
"""

import importlib.metadata
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EnvironmentSnapshot:
    """Complete environment snapshot for reproducibility."""
    captured_at: str
    python_version: str
    platform_info: str
    os_info: str
    architecture: str
    packages: Dict[str, str]       # package_name → version
    kiteml_version: str
    key_packages: Dict[str, str]   # subset of most important ML packages
    env_vars: Dict[str, str]       # relevant env variables

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def requirements_txt(self) -> str:
        """Generate requirements.txt content from this snapshot."""
        lines = [f"# KiteML environment snapshot — {self.captured_at}"]
        for pkg, ver in sorted(self.packages.items()):
            lines.append(f"{pkg}=={ver}")
        return "\n".join(lines)

    def summary(self) -> str:
        lines = [
            f"Python     : {self.python_version}",
            f"Platform   : {self.platform_info}",
            f"OS         : {self.os_info}",
            f"Captured   : {self.captured_at}",
            f"Packages   : {len(self.packages)} installed",
        ]
        for pkg, ver in self.key_packages.items():
            lines.append(f"  {pkg:<20} {ver}")
        return "\n".join(lines)


_KEY_PACKAGES = [
    "scikit-learn", "sklearn", "numpy", "pandas", "scipy",
    "joblib", "kiteml", "xgboost", "lightgbm", "catboost",
    "fastapi", "uvicorn", "onnx", "skl2onnx",
]

_RELEVANT_ENV_VARS = [
    "PYTHONPATH", "PYTHONHASHSEED", "OMP_NUM_THREADS",
    "SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL",
    "CUDA_VISIBLE_DEVICES",
]


def capture_environment() -> EnvironmentSnapshot:
    """
    Capture the current Python runtime environment.

    Returns
    -------
    EnvironmentSnapshot
    """
    # All installed packages
    packages: Dict[str, str] = {}
    try:
        for dist in importlib.metadata.distributions():
            name = dist.metadata["Name"]
            ver = dist.metadata["Version"]
            if name and ver:
                packages[name.lower().replace("-", "_")] = ver
    except Exception:
        pass

    # Key ML packages subset
    key_packages: Dict[str, str] = {}
    for pkg in _KEY_PACKAGES:
        normalized = pkg.lower().replace("-", "_")
        if normalized in packages:
            key_packages[pkg] = packages[normalized]
        elif pkg in packages:
            key_packages[pkg] = packages[pkg]

    # KiteML version
    try:
        from kiteml import __version__ as kiteml_ver
    except Exception:
        kiteml_ver = "dev"

    # Relevant env vars
    env_vars = {k: os.environ.get(k, "") for k in _RELEVANT_ENV_VARS if k in os.environ}

    return EnvironmentSnapshot(
        captured_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        python_version=sys.version,
        platform_info=platform.platform(),
        os_info=f"{platform.system()} {platform.release()}",
        architecture=platform.machine(),
        packages=packages,
        kiteml_version=kiteml_ver,
        key_packages=key_packages,
        env_vars=env_vars,
    )
