"""
tracker.py — ExperimentTracker for tracking experiment metadata and reproducibility parameters.
"""

import hashlib
import platform
import time
from typing import Any

from kiteml.training.context import TrainingContext


class ExperimentTracker:
    """
    Captures dataset fingerprints, system hardware info, and reproducibility settings.
    """

    def track_experiment(self, context: TrainingContext, random_state: int = 42) -> dict[str, Any]:
        """
        Track experiment configuration and environment details.

        Parameters
        ----------
        context : TrainingContext
            Current training context.
        random_state : int
            Random seed.

        Returns
        -------
        dict[str, Any]
            Experiment metadata dictionary.
        """
        df = context.dataset
        dataset_hash = ""
        if df is not None:
            raw_bytes = str(df.shape).encode("utf-8") + str(list(df.columns)).encode("utf-8")
            dataset_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]

        return {
            "timestamp": time.time(),
            "random_state": random_state,
            "dataset_fingerprint": dataset_hash,
            "system_info": {
                "os": platform.system(),
                "python_version": platform.python_version(),
            },
            "task_type": context.task_type,
            "target_name": context.target_name,
        }
