"""
report.py — DeploymentReport data model and terminal summary formatting for KiteML.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DeploymentReport:
    """
    Report container summarizing package creation and validation details.
    """

    package_path: str
    is_valid: bool
    model_name: str
    task_type: str
    file_size_kb: float

    def summary(self, width: int = 55) -> str:
        """Render terminal summary box."""
        lines = [
            "═" * width,
            "📦 KiteML Package & Deployment Report",
            "═" * width,
            f"  Package File  {Path(self.package_path).name}",
            f"  Model Name    {self.model_name}",
            f"  Task Type     {self.task_type}",
            f"  Size          {self.file_size_kb:.2f} KB",
            f"  Integrity     {'VALID ✓' if self.is_valid else 'CORRUPTED ✗'}",
            "═" * width,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to dictionary."""
        return {
            "package_path": self.package_path,
            "is_valid": self.is_valid,
            "model_name": self.model_name,
            "task_type": self.task_type,
            "file_size_kb": self.file_size_kb,
        }

    def to_json(self) -> str:
        """Export report to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
