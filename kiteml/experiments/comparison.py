"""
comparison.py — Compare multiple KiteML experiment runs.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from kiteml.experiments.tracker import ExperimentRun


@dataclass
class RunComparison:
    """Side-by-side comparison of multiple runs."""
    runs: List[ExperimentRun]
    best_run: ExperimentRun
    metric_name: str
    ranked: List[ExperimentRun]     # runs sorted best→worst by metric

    def print_table(self) -> None:
        W = 70
        print("\n" + "═" * W)
        print("  📊  KiteML — Experiment Run Comparison")
        print("═" * W)
        print(f"  {'Run ID':<10} {'Model':<28} {'Score':>8} {'Time(s)':>8} {'Features':>8}")
        print("─" * W)
        for i, run in enumerate(self.ranked):
            score_str = f"{run.score:.4f}" if run.score is not None else "N/A"
            marker = " ★" if i == 0 else "  "
            print(f"{marker} {run.run_id:<10} {run.model_name:<28} {score_str:>8} "
                  f"{run.training_time_s:>8.2f} {run.n_features:>8}")
        print("═" * W)
        print(f"  Best: {self.best_run.run_id} ({self.best_run.model_name})")


def compare_runs(
    runs: List[ExperimentRun],
    metric: str = "score",
    higher_is_better: bool = True,
) -> RunComparison:
    """
    Compare multiple ExperimentRun objects.

    Parameters
    ----------
    runs : list of ExperimentRun
    metric : str
        Attribute to rank by. Default ``'score'``.
    higher_is_better : bool
        Sort direction. Default True.

    Returns
    -------
    RunComparison
    """
    def _get_metric(run: ExperimentRun) -> float:
        val = getattr(run, metric, None)
        if val is None:
            val = run.metrics.get(metric)
        return float(val) if val is not None else float("-inf") if higher_is_better else float("inf")

    ranked = sorted(runs, key=_get_metric, reverse=higher_is_better)
    best = ranked[0]
    return RunComparison(runs=runs, best_run=best, metric_name=metric, ranked=ranked)
