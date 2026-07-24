"""
explainability.py — SelectionExplainer decision rationale generator for KiteML model selection.
"""


class SelectionExplainer:
    """
    Generates transparent human-readable explanations detailing model selection decisions.
    """

    def explain_selection(
        self,
        winner_name: str,
        winner_score: float,
        runner_up_name: str | None = None,
        runner_up_score: float | None = None,
        policy: str = "balanced",
    ) -> str:
        """
        Generate selection rationale.

        Parameters
        ----------
        winner_name : str
            Winning model name.
        winner_score : float
            Winning model composite score.
        runner_up_name : str, optional
            Runner-up model name.
        runner_up_score : float, optional
            Runner-up model composite score.
        policy : str
            Selection policy used.

        Returns
        -------
        str
            Human-readable explanation.
        """
        reasons = [
            f"Selected '{winner_name}' as the optimal model under the '{policy}' policy with a Composite Score of {winner_score:.2f}/100.",
        ]

        if runner_up_name and runner_up_score is not None:
            diff = winner_score - runner_up_score
            reasons.append(
                f"Outperformed runner-up '{runner_up_name}' ({runner_up_score:.2f}/100) by +{diff:.2f} points due to superior accuracy-efficiency trade-offs."
            )
        else:
            reasons.append("Achieved highest composite rating across all candidate models.")

        return " ".join(reasons)
