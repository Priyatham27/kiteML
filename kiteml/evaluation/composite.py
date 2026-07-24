"""
composite.py — CompositeScorer Flagship Composite Score 0-100 rating system for Story 5.4.
"""

from typing import Any

from kiteml.evaluation.benchmark import BenchmarkMetrics


class CompositeScorer:
    """
    Calculates a normalized 0-100 Composite Score balancing predictive accuracy and computational efficiency.
    """

    def calculate_composite_score(
        self,
        metrics: dict[str, Any],
        benchmark: BenchmarkMetrics,
        task_type: str = "classification",
    ) -> float:
        """
        Compute composite score (0.0 to 100.0).

        Parameters
        ----------
        metrics : dict[str, Any]
            Performance metrics.
        benchmark : BenchmarkMetrics
            Benchmark metrics.
        task_type : str
            ML task type.

        Returns
        -------
        float
            Normalized composite score.
        """
        if "regression" in task_type:
            r2 = metrics.get("r2", 0.0)
            perf_score = max(0.0, min(100.0, r2 * 100.0))
        else:
            f1 = metrics.get("f1", metrics.get("accuracy", 0.0))
            perf_score = max(0.0, min(100.0, f1 * 100.0))

        # Latency efficiency score (penalize if latency > 10ms)
        lat = benchmark.inference_latency_ms
        speed_score = max(50.0, 100.0 - min(50.0, lat * 2.0))

        # Weighted combination: 80% performance, 20% efficiency
        composite = (0.80 * perf_score) + (0.20 * speed_score)
        return round(float(composite), 2)
