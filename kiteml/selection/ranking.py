"""
ranking.py — RankingEngine for candidate model evaluation scoring and sorting in KiteML.
"""

from typing import Any

from kiteml.selection.policies import SelectionPolicy


class RankingEngine:
    """
    Ranks candidate model evaluation records using selection policy weights.
    """

    def __init__(self) -> None:
        self.policy_engine = SelectionPolicy()

    def rank_candidates(
        self,
        candidate_evaluations: list[dict[str, Any]],
        policy: str = "balanced",
    ) -> list[dict[str, Any]]:
        """
        Rank candidate model records.

        Parameters
        ----------
        candidate_evaluations : list[dict[str, Any]]
            Candidate model records containing name, model, composite_score, metrics, benchmark.
        policy : str
            Active selection policy profile.

        Returns
        -------
        list[dict[str, Any]]
            List of candidate model records sorted by rank (rank 1 = winner).
        """
        if not candidate_evaluations:
            return []

        weights = self.policy_engine.get_weights(policy)

        scored_candidates = []
        for cand in candidate_evaluations:
            c_score = cand.get("composite_score", 0.0)
            lat = cand.get("benchmark", {}).get("inference_latency_ms", 1.0)
            speed_factor = max(0.0, 100.0 - (lat * 2.0))

            weighted_score = (weights["performance"] * c_score) + (weights["speed"] * speed_factor)
            cand_copy = dict(cand)
            cand_copy["policy_score"] = round(weighted_score, 2)
            scored_candidates.append(cand_copy)

        scored_candidates.sort(key=lambda c: c["policy_score"], reverse=True)

        for idx, cand in enumerate(scored_candidates, 1):
            cand["rank"] = idx

        return scored_candidates
