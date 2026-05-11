"""
run_manager.py — Load, list, and manage KiteML experiment runs.

Provides utilities to list, load, delete, and export experiment runs
stored by experiments/tracker.py.
"""

import json
import os
import shutil
from typing import Optional

from kiteml.experiments.tracker import _DEFAULT_STORE, ExperimentRun


def list_experiments(store_path: Optional[str] = None) -> list[str]:
    """Return all experiment names in the store."""
    store = store_path or _DEFAULT_STORE
    if not os.path.isdir(store):
        return []
    return [d for d in os.listdir(store) if os.path.isdir(os.path.join(store, d))]


def load_run(run_id: str, experiment_name: str = "default", store_path: Optional[str] = None) -> ExperimentRun:
    """Load a single ExperimentRun by ID."""
    store = store_path or _DEFAULT_STORE
    run_file = os.path.join(store, experiment_name, f"{run_id}.json")
    if not os.path.exists(run_file):
        raise FileNotFoundError(f"Run '{run_id}' not found in experiment '{experiment_name}'.")
    with open(run_file, encoding="utf-8") as f:
        data = json.load(f)
    return ExperimentRun(**data)


def delete_run(run_id: str, experiment_name: str = "default", store_path: Optional[str] = None) -> None:
    """Delete a single experiment run."""
    store = store_path or _DEFAULT_STORE
    run_file = os.path.join(store, experiment_name, f"{run_id}.json")
    if os.path.exists(run_file):
        os.remove(run_file)
        print(f"🗑️  Deleted run {run_id} from '{experiment_name}'.")
    else:
        raise FileNotFoundError(f"Run '{run_id}' not found.")


def delete_experiment(experiment_name: str, store_path: Optional[str] = None) -> None:
    """Delete all runs for an experiment."""
    store = store_path or _DEFAULT_STORE
    exp_dir = os.path.join(store, experiment_name)
    if os.path.isdir(exp_dir):
        shutil.rmtree(exp_dir)
        print(f"🗑️  Deleted experiment '{experiment_name}'.")
    else:
        raise FileNotFoundError(f"Experiment '{experiment_name}' not found.")


def best_run(
    experiment_name: str = "default",
    metric: str = "score",
    higher_is_better: bool = True,
    store_path: Optional[str] = None,
) -> Optional[ExperimentRun]:
    """Return the best ExperimentRun for an experiment."""
    from kiteml.experiments.tracker import list_runs

    runs = list_runs(experiment_name, store_path=store_path)
    if not runs:
        return None

    def _get(r: ExperimentRun) -> float:
        val = getattr(r, metric, None) or r.metrics.get(metric)
        return float(val) if val is not None else float("-inf")

    return sorted(runs, key=_get, reverse=higher_is_better)[0]


def export_runs_csv(
    experiment_name: str = "default",
    output_path: Optional[str] = None,
    store_path: Optional[str] = None,
) -> str:
    """Export all runs for an experiment to a CSV file."""
    import csv

    from kiteml.experiments.tracker import list_runs

    runs = list_runs(experiment_name, store_path=store_path)
    if not runs:
        print("⚠️  No runs to export.")
        return ""

    out = output_path or f"{experiment_name}_runs.csv"
    fieldnames = [
        "run_id",
        "model_name",
        "problem_type",
        "score",
        "training_time_s",
        "n_features",
        "created_at",
        "notes",
    ]

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in runs:
            writer.writerow(r.to_dict())

    print(f"📄 Exported {len(runs)} runs → {out}")
    return out
