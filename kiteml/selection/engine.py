"""
engine.py — ModelSelectionEngine master entry point for KiteML algorithm selection.
"""

from typing import Any

from kiteml.selection.best_model import BestModel
from kiteml.selection.explainability import SelectionExplainer
from kiteml.selection.pareto import ParetoSelector
from kiteml.selection.ranking import RankingEngine
from kiteml.selection.report import SelectionReport


class ModelSelectionEngine:
    """
    Master Intelligent Model Selection Engine selecting optimal algorithms across candidate models.
    """

    def __init__(self) -> None:
        self.ranking_engine = RankingEngine()
        self.pareto_selector = ParetoSelector()
        self.explainer = SelectionExplainer()

    def select(
        self,
        candidate_evaluations: list[dict[str, Any]],
        policy: str = "balanced",
    ) -> BestModel:
        """
        Select optimal model algorithm across candidate evaluation records.

        Parameters
        ----------
        candidate_evaluations : list[dict[str, Any]]
            List of candidate evaluation dictionaries containing name, model, composite_score, metrics, etc.
        policy : str
            Active selection policy profile ('balanced', 'accuracy', 'fast_inference', 'low_memory').

        Returns
        -------
        BestModel
            Selected optimal BestModel instance.
        """
        if not candidate_evaluations:
            raise ValueError("Cannot select best model from empty candidate evaluation list.")

        ranked = self.ranking_engine.rank_candidates(candidate_evaluations, policy=policy)
        pareto = self.pareto_selector.find_pareto_frontier(candidate_evaluations)

        winner_record = ranked[0]
        runner_up_record = ranked[1] if len(ranked) > 1 else None

        explanation = self.explainer.explain_selection(
            winner_name=winner_record["name"],
            winner_score=winner_record["policy_score"],
            runner_up_name=runner_up_record["name"] if runner_up_record else None,
            runner_up_score=runner_up_record["policy_score"] if runner_up_record else None,
            policy=policy,
        )

        best_model = BestModel(
            model=winner_record["model"],
            model_name=winner_record["name"],
            pipeline=winner_record.get("pipeline"),
            evaluation=winner_record.get("evaluation"),
            composite_score=winner_record["policy_score"],
            explanation=explanation,
            pareto_frontier=pareto,
        )

        return best_model

    def generate_report(
        self,
        candidate_evaluations: list[dict[str, Any]],
        policy: str = "balanced",
    ) -> SelectionReport:
        """
        Generate complete SelectionReport with leaderboard.

        Parameters
        ----------
        candidate_evaluations : list[dict[str, Any]]
            Candidate model records.
        policy : str
            Active selection policy profile.

        Returns
        -------
        SelectionReport
            Full selection report with leaderboard and Pareto frontier.
        """
        best_model = self.select(candidate_evaluations, policy=policy)
        ranked = self.ranking_engine.rank_candidates(candidate_evaluations, policy=policy)
        pareto = self.pareto_selector.find_pareto_frontier(candidate_evaluations)

        return SelectionReport(
            best_model=best_model,
            leaderboard=ranked,
            pareto_frontier=pareto,
            policy=policy,
        )
