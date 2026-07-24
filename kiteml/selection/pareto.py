"""
pareto.py — ParetoSelector Flagship Multi-Objective Pareto Selection Feature for Story 5.5.
"""

from typing import Any


class ParetoSelector:
    """
    Identifies non-dominated candidate models along the multi-objective Pareto frontier.
    """

    def find_pareto_frontier(self, candidate_evaluations: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Find Pareto frontier non-dominated models.

        Parameters
        ----------
        candidate_evaluations : list[dict[str, Any]]
            List of candidate evaluation dictionaries containing name, composite_score, metrics, benchmark.

        Returns
        -------
        dict[str, Any]
            Pareto frontier summary (best_overall, fastest, most_efficient, most_accurate).
        """
        if not candidate_evaluations:
            return {}

        best_overall = max(candidate_evaluations, key=lambda c: c.get("composite_score", 0.0))
        fastest = min(
            candidate_evaluations,
            key=lambda c: c.get("benchmark", {}).get("inference_latency_ms", 999.0),
        )
        most_efficient = min(
            candidate_evaluations,
            key=lambda c: c.get("benchmark", {}).get("model_size_kb", 999999.0),
        )

        def get_primary_metric(c: dict[str, Any]) -> float:
            m = c.get("metrics", {})
            return float(m.get("f1", m.get("accuracy", m.get("r2", 0.0))))

        most_accurate = max(candidate_evaluations, key=get_primary_metric)

        return {
            "best_overall": best_overall.get("name"),
            "fastest_inference": fastest.get("name"),
            "most_efficient_memory": most_efficient.get("name"),
            "most_accurate": most_accurate.get("name"),
        }
