"""
reproducibility.py — Capture full reproducibility snapshots for KiteML runs.

Ensures every training run can be exactly recreated: seeds, packages, OS,
Python version, and configuration.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict

from kiteml.config import DEFAULT_CV_FOLDS, DEFAULT_RANDOM_STATE
from kiteml.deployment.environment_capture import capture_environment


@dataclass
class ReproducibilitySnapshot:
    """Complete snapshot for run reproducibility."""

    run_id: str
    random_seed: int
    cv_folds: int
    python_version: str
    platform: str
    key_packages: Dict[str, str]
    kiteml_config: Dict[str, Any]
    env_hash: str
    created_at: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def matches(self, other: "ReproducibilitySnapshot") -> bool:
        """Check if two snapshots have matching environments."""
        return (
            self.random_seed == other.random_seed
            and self.python_version.split()[0] == other.python_version.split()[0]
            and self.key_packages == other.key_packages
        )


def capture_snapshot(run_id: str = "unknown") -> ReproducibilitySnapshot:
    """
    Capture a complete reproducibility snapshot of the current environment.

    Parameters
    ----------
    run_id : str
        The training run ID to associate with this snapshot.

    Returns
    -------
    ReproducibilitySnapshot
    """
    env = capture_environment()

    from kiteml import config as cfg

    kiteml_config = {
        k: v for k, v in cfg.__dict__.items() if not k.startswith("_") and isinstance(v, (int, float, str, bool))
    }

    # Compute env hash (key packages + python version)
    env_str = json.dumps(
        {"python": env.python_version, "packages": env.key_packages},
        sort_keys=True,
    )
    env_hash = hashlib.md5(env_str.encode()).hexdigest()[:12]

    return ReproducibilitySnapshot(
        run_id=run_id,
        random_seed=DEFAULT_RANDOM_STATE,
        cv_folds=DEFAULT_CV_FOLDS,
        python_version=env.python_version,
        platform=env.platform_info,
        key_packages=env.key_packages,
        kiteml_config=kiteml_config,
        env_hash=env_hash,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
