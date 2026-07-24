"""
manifest.py — PipelineManifest model for KiteML pipeline serialization.
"""

import datetime
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

import kiteml


@dataclass
class PipelineManifest:
    """
    Manifest describing metadata, version info, and stage structure of a serialized .kml pipeline package.
    """

    kiteml_version: str = field(default_factory=lambda: getattr(kiteml, "__version__", "1.0.0"))
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    pipeline_version: str = "1.0"
    checksum: str = ""
    feature_count: int = 0
    target_name: str | None = None
    stage_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize manifest to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize manifest to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PipelineManifest":
        """Reconstruct PipelineManifest from dictionary."""
        return cls(
            kiteml_version=d.get("kiteml_version", "1.0.0"),
            python_version=d.get("python_version", sys.version.split()[0]),
            created_at=d.get("created_at", ""),
            pipeline_version=d.get("pipeline_version", "1.0"),
            checksum=d.get("checksum", ""),
            feature_count=d.get("feature_count", 0),
            target_name=d.get("target_name"),
            stage_names=d.get("stage_names", []),
        )

    @classmethod
    def from_json(cls, s: str) -> "PipelineManifest":
        """Reconstruct PipelineManifest from JSON string."""
        return cls.from_dict(json.loads(s))
