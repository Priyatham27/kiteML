"""
manifest.py — Generate YAML/JSON deployment manifests for KiteML bundles.

A manifest is the "passport" of a deployed model — it describes what the model
is, what it expects, and how to use it safely.
"""

import contextlib
import json
import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ModelManifest:
    """Complete deployment manifest for a KiteML model bundle."""

    bundle_id: str
    model_name: str
    problem_type: str
    kiteml_version: str
    created_at: str
    python_version: str

    # Model metadata
    score: Optional[float]
    metric_name: str
    feature_names: list[str]
    n_features: int

    # Schema
    input_schema: dict[str, str]  # feature → dtype
    target_column: Optional[str]

    # Artifacts
    artifacts: dict[str, str]  # artifact_name → relative path
    checksums: dict[str, str]  # artifact_name → MD5

    # Reproduction
    random_seed: Optional[int]
    training_duration_s: Optional[float]
    notes: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def to_yaml(self) -> str:
        """Render as YAML-formatted string (no external dep)."""
        d = self.to_dict()
        lines = ["# KiteML Model Manifest", f"# Generated: {self.created_at}", "---"]

        def _render(obj, indent=0):
            pad = "  " * indent
            out = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (dict, list)):
                        out.append(f"{pad}{k}:")
                        out.extend(_render(v, indent + 1))
                    else:
                        out.append(f"{pad}{k}: {json.dumps(v)}")
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        out.append(f"{pad}-")
                        out.extend(_render(item, indent + 1))
                    else:
                        out.append(f"{pad}- {json.dumps(item)}")
            return out

        lines.extend(_render(d))
        return "\n".join(lines)

    def save(self, path: str) -> None:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_yaml())

    @classmethod
    def load(cls, path: str) -> "ModelManifest":
        """Load manifest from JSON (YAML saved as JSON-compatible)."""
        with open(path, encoding="utf-8") as f:
            f.read()
        # Parse the pseudo-YAML (simple key: value)
        raise NotImplementedError("Use manifest.json (load_json) for programmatic loading.")


def build_manifest(
    result: Any,
    bundle_id: str,
    artifacts: dict[str, str],
    checksums: dict[str, str],
    target_column: Optional[str] = None,
) -> ModelManifest:
    """
    Build a ModelManifest from a KiteML Result object.

    Parameters
    ----------
    result : Result
        Fitted KiteML result.
    bundle_id : str
        Unique identifier for this bundle (e.g. UUID or timestamp).
    artifacts : dict
        Map of artifact name → relative file path within the bundle.
    checksums : dict
        Map of artifact name → MD5 checksum.
    target_column : str, optional

    Returns
    -------
    ModelManifest
    """
    import sys

    from kiteml.config import DEFAULT_RANDOM_STATE

    try:
        from kiteml import __version__ as kiteml_ver
    except Exception:
        kiteml_ver = "dev"

    # Build input schema
    input_schema: dict[str, str] = {}
    if result.preprocessor is not None:
        try:
            for feat in result.feature_names or []:
                input_schema[feat] = "numeric"
        except Exception:
            pass

    # Get score
    score = None
    with contextlib.suppress(Exception):
        score = float(result.score)

    # Metric name
    metric_name = "accuracy" if result.problem_type == "classification" else "r2"

    return ModelManifest(
        bundle_id=bundle_id,
        model_name=result.model_name,
        problem_type=result.problem_type,
        kiteml_version=kiteml_ver,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        python_version=sys.version.split()[0],
        score=score,
        metric_name=metric_name,
        feature_names=list(result.feature_names or []),
        n_features=len(result.feature_names or []),
        input_schema=input_schema,
        target_column=target_column,
        artifacts=artifacts,
        checksums=checksums,
        random_seed=DEFAULT_RANDOM_STATE,
        training_duration_s=round(result.times.total, 3) if result.times else None,
    )
