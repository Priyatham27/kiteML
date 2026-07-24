"""
pickle.py — PickleAdapter for exporting serialized pickle model files in KiteML.
"""

import pickle
from pathlib import Path
from typing import Any

from kiteml.deployment.adapters import DeploymentAdapter


class PickleAdapter(DeploymentAdapter):
    """
    Exports standalone serialized pickle model files.
    """

    @property
    def adapter_name(self) -> str:
        return "pickle"

    def export(
        self,
        model: Any,
        output_dir: Path | str,
        pipeline: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Export pickle model file."""
        out_path = Path(output_dir)
        if out_path.is_dir() or not out_path.name.endswith(".pkl"):
            out_path.mkdir(parents=True, exist_ok=True)
            out_file = out_path / "model.pkl"
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_file = out_path

        with open(out_file, "wb") as f:
            pickle.dump({"model": model, "pipeline": pipeline}, f)
        return out_file
