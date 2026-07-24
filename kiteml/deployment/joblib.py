"""
joblib.py — JoblibAdapter for exporting serialized joblib estimator files in KiteML.
"""

from pathlib import Path
from typing import Any

import joblib

from kiteml.deployment.adapters import DeploymentAdapter


class JoblibAdapter(DeploymentAdapter):
    """
    Exports standalone serialized joblib model files.
    """

    @property
    def adapter_name(self) -> str:
        return "joblib"

    def export(
        self,
        model: Any,
        output_dir: Path | str,
        pipeline: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """
        Export joblib model file.

        Parameters
        ----------
        model : Any
            Fitted estimator model.
        output_dir : Path | str
            Target output directory or file path.
        pipeline : Any, optional
            Fitted pipeline instance.
        metadata : dict[str, Any], optional
            Export metadata.

        Returns
        -------
        Path
            Path to exported model.joblib file.
        """
        out_path = Path(output_dir)
        if out_path.is_dir() or not out_path.name.endswith(".joblib"):
            out_path.mkdir(parents=True, exist_ok=True)
            out_file = out_path / "model.joblib"
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_file = out_path

        joblib.dump({"model": model, "pipeline": pipeline}, out_file)
        return out_file
