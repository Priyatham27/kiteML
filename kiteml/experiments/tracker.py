"""
tracker.py — Experiment tracking for KiteML training runs.

Automatically captures run metadata from a Result object and stores it
in a local JSON-based experiment store.  No external server needed.
"""

import contextlib
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

_DEFAULT_STORE = os.path.join(os.path.expanduser("~"), ".kiteml", "experiments")


@dataclass
class ExperimentRun:
    """A single recorded training run."""

    run_id: str
    experiment_name: str
    model_name: str
    problem_type: str
    score: Optional[float]
    metrics: dict[str, float]
    training_time_s: float
    n_features: int
    feature_names: list[str]
    all_results: dict[str, float]
    dataset_hash: Optional[str]
    tags: dict[str, str]
    notes: str
    created_at: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def summary(self) -> str:
        score_str = f"{self.score:.4f}" if self.score is not None else "N/A"
        return (
            f"Run: {self.run_id}  |  Model: {self.model_name}  |  "
            f"Score: {score_str}  |  Time: {self.training_time_s:.2f}s  |  "
            f"Created: {self.created_at}"
        )


def _hash_dataframe(df: Any) -> str:
    """Compute a short hash of a DataFrame for reproducibility."""
    try:
        import pandas as pd

        h = hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()
        return h[:12]
    except Exception:
        return "unknown"


def track(
    result: Any,
    experiment_name: str = "default",
    dataset: Optional[Any] = None,
    tags: Optional[dict[str, str]] = None,
    notes: str = "",
    store_path: Optional[str] = None,
) -> ExperimentRun:
    """
    Record a KiteML training run to the experiment store.

    Parameters
    ----------
    result : Result
        Fitted KiteML result.
    experiment_name : str
        Logical group name for this set of runs.
    dataset : pd.DataFrame, optional
        Training dataset (used for hash only).
    tags : dict, optional
        Free-form key-value metadata.
    notes : str
        Human-readable notes.
    store_path : str, optional
        Override default experiment store directory.

    Returns
    -------
    ExperimentRun
    """
    run_id = str(uuid.uuid4())[:8]
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dataset_hash = _hash_dataframe(dataset) if dataset is not None else None

    # Extract metrics from typed dataclass
    metrics: dict[str, float] = {}
    try:
        if hasattr(result.metrics, "__dict__"):
            for k, v in result.metrics.__dict__.items():
                with contextlib.suppress(TypeError, ValueError):
                    metrics[k] = float(v)
    except Exception:
        pass

    score = None
    with contextlib.suppress(Exception):
        score = float(result.score)

    all_results: dict[str, float] = {}
    for k, v in (result.all_results or {}).items():
        with contextlib.suppress(Exception):
            all_results[k] = float(v)

    run = ExperimentRun(
        run_id=run_id,
        experiment_name=experiment_name,
        model_name=result.model_name,
        problem_type=result.problem_type,
        score=score,
        metrics=metrics,
        training_time_s=round(result.times.total, 3) if result.times else 0.0,
        n_features=len(result.feature_names or []),
        feature_names=list(result.feature_names or []),
        all_results=all_results,
        dataset_hash=dataset_hash,
        tags=tags or {},
        notes=notes,
        created_at=created_at,
    )

    # Persist to local store
    store = store_path or _DEFAULT_STORE
    exp_dir = os.path.join(store, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    run_file = os.path.join(exp_dir, f"{run_id}.json")
    with open(run_file, "w", encoding="utf-8") as f:
        json.dump(run.to_dict(), f, indent=2, default=str)

    print(f"📊 Experiment run tracked → {run_file}")
    print(f"   {run.summary()}")
    return run


def list_runs(
    experiment_name: str = "default",
    store_path: Optional[str] = None,
) -> list[ExperimentRun]:
    """List all recorded runs for an experiment."""
    store = store_path or _DEFAULT_STORE
    exp_dir = os.path.join(store, experiment_name)
    if not os.path.isdir(exp_dir):
        return []

    runs: list[ExperimentRun] = []
    for fname in sorted(os.listdir(exp_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(exp_dir, fname), encoding="utf-8") as f:
                data = json.load(f)
            runs.append(ExperimentRun(**data))

    return runs
