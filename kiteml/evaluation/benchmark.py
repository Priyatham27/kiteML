"""
benchmark.py — BenchmarkEngine for evaluating inference speed and memory footprint in KiteML.
"""

import pickle
import time
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass
class BenchmarkMetrics:
    """
    Performance and computational footprint benchmark metrics.
    """

    inference_latency_ms: float = 0.0
    total_predict_time_sec: float = 0.0
    model_size_kb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize benchmark metrics to dictionary."""
        return asdict(self)


class BenchmarkEngine:
    """
    Measures inference latency and model memory size.
    """

    def benchmark_model(self, model: Any, X_test: pd.DataFrame | Any) -> BenchmarkMetrics:
        """
        Run inference benchmarks on fitted model.

        Parameters
        ----------
        model : Any
            Fitted estimator instance.
        X_test : pd.DataFrame | Any
            Test feature matrix.

        Returns
        -------
        BenchmarkMetrics
            Recorded benchmark metrics.
        """
        n_samples = len(X_test) if hasattr(X_test, "__len__") else 1

        t0 = time.time()
        if hasattr(model, "predict"):
            model.predict(X_test)
        elapsed = time.time() - t0

        latency_ms = (elapsed / max(1, n_samples)) * 1000.0

        size_kb = 0.0
        try:
            raw_bytes = pickle.dumps(model)
            size_kb = len(raw_bytes) / 1024.0
        except Exception:
            size_kb = 0.0

        return BenchmarkMetrics(
            inference_latency_ms=latency_ms,
            total_predict_time_sec=elapsed,
            model_size_kb=size_kb,
        )
