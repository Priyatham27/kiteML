"""
packaging.py — Complete deployable bundle packaging for KiteML.

Creates a self-contained .kiteml bundle directory containing all artifacts
needed for production deployment: model, preprocessor, schema, metadata.

Bundle Layout
-------------
<name>.kiteml/
├── model.joblib          ← fitted sklearn estimator
├── preprocessor.joblib   ← fitted Preprocessor (with sklearn Pipeline)
├── schema.json           ← feature schema for validation
├── metrics.json          ← evaluation metrics
├── metadata.json         ← full result metadata
├── manifest.yaml         ← deployment manifest (YAML)
├── environment.txt       ← requirements.txt snapshot
└── lineage.json          ← training lineage
"""

import contextlib
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from kiteml.deployment.environment_capture import capture_environment
from kiteml.deployment.manifest import build_manifest
from kiteml.deployment.serialization import _md5, save_joblib, save_json


@dataclass
class PackageResult:
    """Result of a packaging operation."""

    bundle_path: str
    bundle_id: str
    artifacts: dict[str, str]
    total_size_bytes: int
    created_at: str

    def __str__(self) -> str:
        size_mb = self.total_size_bytes / 1e6
        return (
            f"📦 KiteML Bundle\n"
            f"   Path      : {self.bundle_path}\n"
            f"   Bundle ID : {self.bundle_id}\n"
            f"   Size      : {size_mb:.2f} MB\n"
            f"   Created   : {self.created_at}\n"
            f"   Artifacts : {list(self.artifacts.keys())}"
        )


def package(
    result: Any,
    path: str,
    target_column: Optional[str] = None,
    overwrite: bool = False,
) -> PackageResult:
    """
    Package a KiteML Result into a self-contained deployable bundle.

    Parameters
    ----------
    result : Result
        Fitted KiteML result object.
    path : str
        Output directory path. Conventionally ends in ``.kiteml``
        (e.g. ``'customer_churn.kiteml'``).
    target_column : str, optional
        Name of the training target column (for metadata).
    overwrite : bool
        If True, overwrite an existing bundle at this path.

    Returns
    -------
    PackageResult
    """
    if os.path.exists(path):
        if overwrite:
            shutil.rmtree(path)
        else:
            raise FileExistsError(f"Bundle already exists at '{path}'. Use overwrite=True to replace.")

    os.makedirs(path, exist_ok=True)
    bundle_id = str(uuid.uuid4())[:8]
    artifacts: dict[str, str] = {}
    checksums: dict[str, str] = {}

    # ── 1. Model ──────────────────────────────────────────────────────────
    model_path = os.path.join(path, "model.joblib")
    r = save_joblib(result.model, model_path)
    artifacts["model"] = "model.joblib"
    checksums["model"] = r.checksum

    # ── 2. Preprocessor ───────────────────────────────────────────────────
    if result.preprocessor is not None:
        prep_path = os.path.join(path, "preprocessor.joblib")
        r2 = save_joblib(result.preprocessor, prep_path)
        artifacts["preprocessor"] = "preprocessor.joblib"
        checksums["preprocessor"] = r2.checksum

    # ── 3. Schema ─────────────────────────────────────────────────────────
    schema = _build_schema(result)
    schema_path = os.path.join(path, "schema.json")
    save_json(schema, schema_path)
    artifacts["schema"] = "schema.json"
    checksums["schema"] = _md5(schema_path)

    # ── 4. Metrics ────────────────────────────────────────────────────────
    metrics_dict = _metrics_to_dict(result.metrics)
    metrics_path = os.path.join(path, "metrics.json")
    save_json(metrics_dict, metrics_path)
    artifacts["metrics"] = "metrics.json"
    checksums["metrics"] = _md5(metrics_path)

    # ── 5. Metadata ───────────────────────────────────────────────────────
    metadata = {
        "bundle_id": bundle_id,
        "model_name": result.model_name,
        "problem_type": result.problem_type,
        "feature_names": list(result.feature_names or []),
        "n_features": len(result.feature_names or []),
        "target_column": target_column,
        "training_time_s": round(result.times.total, 3) if result.times else None,
        "score": _safe_float(result.score),
        "all_results": {k: _safe_float(v) for k, v in (result.all_results or {}).items()},
        "packaged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = os.path.join(path, "metadata.json")
    save_json(metadata, meta_path)
    artifacts["metadata"] = "metadata.json"
    checksums["metadata"] = _md5(meta_path)

    # ── 6. Manifest (YAML) ────────────────────────────────────────────────
    manifest = build_manifest(result, bundle_id, artifacts, checksums, target_column)
    manifest_path = os.path.join(path, "manifest.yaml")
    manifest.save(manifest_path)
    artifacts["manifest"] = "manifest.yaml"

    # ── 7. Environment snapshot ───────────────────────────────────────────
    env = capture_environment()
    env_path = os.path.join(path, "environment.txt")
    with open(env_path, "w", encoding="utf-8") as f:
        f.write(env.requirements_txt())
    artifacts["environment"] = "environment.txt"

    # ── 8. Feature importances (optional) ─────────────────────────────────
    if result.feature_importances:
        fi_path = os.path.join(path, "feature_importances.json")
        save_json(
            {k: _safe_float(v) for k, v in result.feature_importances.items()},
            fi_path,
        )
        artifacts["feature_importances"] = "feature_importances.json"
        checksums["feature_importances"] = _md5(fi_path)

    # ── Compute total size ────────────────────────────────────────────────
    total_bytes = sum(
        os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
    )

    result_obj = PackageResult(
        bundle_path=os.path.abspath(path),
        bundle_id=bundle_id,
        artifacts=artifacts,
        total_size_bytes=total_bytes,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    print(result_obj)
    return result_obj


def load_bundle(path: str) -> dict[str, Any]:
    """
    Load all artifacts from a .kiteml bundle directory.

    Parameters
    ----------
    path : str
        Path to a .kiteml bundle directory.

    Returns
    -------
    dict with keys: model, preprocessor (optional), schema, metrics, metadata
    """
    from kiteml.deployment.serialization import load_joblib, load_json

    if not os.path.isdir(path):
        raise NotADirectoryError(f"Bundle not found: '{path}'")

    bundle: dict[str, Any] = {}

    model_path = os.path.join(path, "model.joblib")
    if os.path.exists(model_path):
        bundle["model"] = load_joblib(model_path)

    prep_path = os.path.join(path, "preprocessor.joblib")
    if os.path.exists(prep_path):
        bundle["preprocessor"] = load_joblib(prep_path)

    schema_path = os.path.join(path, "schema.json")
    if os.path.exists(schema_path):
        bundle["schema"] = load_json(schema_path)

    metrics_path = os.path.join(path, "metrics.json")
    if os.path.exists(metrics_path):
        bundle["metrics"] = load_json(metrics_path)

    meta_path = os.path.join(path, "metadata.json")
    if os.path.exists(meta_path):
        bundle["metadata"] = load_json(meta_path)

    fi_path = os.path.join(path, "feature_importances.json")
    if os.path.exists(fi_path):
        bundle["feature_importances"] = load_json(fi_path)

    return bundle


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_schema(result: Any) -> dict:
    """Extract feature schema from a Result object."""
    schema: dict[str, Any] = {
        "feature_names": list(result.feature_names or []),
        "n_features": len(result.feature_names or []),
        "problem_type": result.problem_type,
        "model_name": result.model_name,
    }
    # Add data profile schema if available
    if result.data_profile is not None:
        with contextlib.suppress(Exception):
            schema["column_types"] = {
                col: profile.column_type.value for col, profile in result.data_profile.column_analysis.profiles.items()
            }
    return schema


def _metrics_to_dict(metrics: Any) -> dict:
    """Convert typed metrics dataclass or dict to serializable dict."""
    if hasattr(metrics, "__dict__"):
        return {k: _safe_float(v) for k, v in metrics.__dict__.items()}
    if isinstance(metrics, dict):
        return {k: _safe_float(v) for k, v in metrics.items()}
    return {}


def _safe_float(v: Any) -> Any:
    """Convert numpy floats to Python floats for JSON safety."""
    try:
        import numpy as np

        if isinstance(v, (np.floating, np.integer)):
            return float(v)
    except Exception:
        pass
    return v
