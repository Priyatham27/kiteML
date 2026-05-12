"""
versioning.py — Model version management for KiteML.

Manages semantic versioning of ML models with bump logic and
version history tracking in a local store.
"""

import contextlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ModelVersion:
    """A semantic version record for a KiteML model."""

    version: str  # e.g. "v1.2.0"
    model_name: str
    score: float | None
    problem_type: str
    notes: str
    created_at: str
    bundle_path: str | None = None


@dataclass
class VersionRegistry:
    """Registry of all versions for a model."""

    model_name: str
    versions: list[ModelVersion]
    current_version: str | None

    def latest(self) -> ModelVersion | None:
        return self.versions[-1] if self.versions else None


_DEFAULT_REGISTRY = os.path.join(os.path.expanduser("~"), ".kiteml", "versions")


def _parse_version(v: str) -> tuple:
    """Parse 'v1.2.3' → (1, 2, 3)."""
    m = re.match(r"v?(\d+)\.(\d+)\.(\d+)", v)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return (0, 0, 0)


def _bump(version: str, bump_type: str = "patch") -> str:
    """Bump a semantic version string."""
    maj, min_, pat = _parse_version(version)
    if bump_type == "major":
        return f"v{maj+1}.0.0"
    elif bump_type == "minor":
        return f"v{maj}.{min_+1}.0"
    else:
        return f"v{maj}.{min_}.{pat+1}"


def version_model(
    result: Any,
    version: str | None = None,
    bump: str = "patch",
    notes: str = "",
    bundle_path: str | None = None,
    store_path: str | None = None,
) -> ModelVersion:
    """
    Record a semantic version for a trained KiteML model.

    Parameters
    ----------
    result : Result
    version : str, optional
        Explicit version string (e.g. ``'v1.2.0'``). If None, auto-bumps.
    bump : str
        ``'major'``, ``'minor'``, or ``'patch'`` (default) for auto-bump.
    notes : str
        Release notes.
    bundle_path : str, optional
        Path to the associated .kiteml bundle.
    store_path : str, optional
        Override default version store directory.

    Returns
    -------
    ModelVersion
    """
    store = store_path or _DEFAULT_REGISTRY
    model_dir = os.path.join(store, result.model_name.replace(" ", "_"))
    os.makedirs(model_dir, exist_ok=True)

    history_path = os.path.join(model_dir, "history.json")
    history: list[dict] = []
    if os.path.exists(history_path):
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)

    # Determine version
    if version is None:
        if history:
            last_ver = history[-1]["version"]
            version = _bump(last_ver, bump)
        else:
            version = "v1.0.0"

    score = None
    with contextlib.suppress(Exception):
        score = round(float(result.score), 4)

    mv = ModelVersion(
        version=version,
        model_name=result.model_name,
        score=score,
        problem_type=result.problem_type,
        notes=notes,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        bundle_path=bundle_path,
    )

    history.append(mv.__dict__)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"🏷️  Model versioned → {version} ({result.model_name})")
    return mv


def get_history(
    model_name: str,
    store_path: str | None = None,
) -> list[ModelVersion]:
    """Retrieve version history for a model."""
    store = store_path or _DEFAULT_REGISTRY
    history_path = os.path.join(store, model_name.replace(" ", "_"), "history.json")
    if not os.path.exists(history_path):
        return []
    with open(history_path, encoding="utf-8") as f:
        data = json.load(f)
    return [ModelVersion(**d) for d in data]
