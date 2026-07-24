"""
manifest.py — DeploymentManifest metadata data model for .kiteml package tracking.
"""

import datetime
import platform
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DeploymentManifest:
    """
    Metadata manifest embedded inside .kiteml deployment archives.
    """

    model_name: str
    task_type: str
    package_version: str = "1.0.0"
    kiteml_version: str = "1.0.2"
    python_version: str = platform.python_version()
    created_at: str = datetime.datetime.now(datetime.timezone.utc).isoformat()
    checksum: str = ""
    format_identifier: str = "KITEML_PACKAGE_V1"

    def to_dict(self) -> dict[str, Any]:
        """Serialize manifest to dictionary."""
        return asdict(self)

    def save(self, path: str) -> None:
        """Save manifest as json."""
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def build_manifest(
    result: Any,
    bundle_id: str,
    artifacts: Any,
    checksums: Any,
    target_column: Any = None,
) -> DeploymentManifest:
    """Legacy manifest builder helper."""
    model_name = getattr(result, "model_name", "Model")
    problem_type = getattr(result, "problem_type", "classification")
    return DeploymentManifest(model_name=model_name, task_type=problem_type)
