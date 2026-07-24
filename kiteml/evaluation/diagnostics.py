"""
diagnostics.py — DiagnosticsEngine for model error spread analysis in KiteML.
"""

from typing import Any

import numpy as np


class DiagnosticsEngine:
    """
    Generates in-depth statistical diagnostics for model prediction errors.
    """

    def diagnose(
        self,
        y_true: Any,
        y_pred: Any,
        task_type: str = "classification",
    ) -> dict[str, Any]:
        """
        Generate evaluation diagnostics.

        Parameters
        ----------
        y_true : Any
            Ground truth targets.
        y_pred : Any
            Predicted values.
        task_type : str
            ML task type.

        Returns
        -------
        dict[str, Any]
            Diagnostic metrics dictionary.
        """
        y_t = np.asarray(y_true)
        y_p = np.asarray(y_pred)

        if "regression" in task_type:
            residuals = y_t - y_p
            return {
                "residual_mean": float(np.mean(residuals)),
                "residual_std": float(np.std(residuals)),
                "residual_max": float(np.max(np.abs(residuals))),
            }

        # Classification diagnostics
        correct = np.equal(y_t, y_p)
        return {
            "error_rate": float(1.0 - np.mean(correct)),
            "total_samples": len(y_t),
        }
