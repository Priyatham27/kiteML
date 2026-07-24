"""
voting.py — FeatureSelectionVotingSystem ensemble voting aggregator for KiteML.
"""

from typing import Any, Sequence

import pandas as pd

from kiteml.feature_selection.blueprint import FeatureScore
from kiteml.feature_selection.rules import FSRuleEngine
from kiteml.feature_selection.selectors import BaseSelector
from kiteml.feature_selection.strategy import SelectionDecision


class FeatureSelectionVotingSystem:
    """
    Feature Selection Voting System (Flagship Feature for Story 4.3).

    Aggregates votes from multiple specialized selectors using weighted consensus,
    producing robust, explainable feature selection decisions.
    """

    def evaluate_feature(
        self,
        col: str,
        df: pd.DataFrame,
        selectors: Sequence[BaseSelector],
        rules: FSRuleEngine,
        data_profile: Any = None,
        target: str | None = None,
        protected_features: Sequence[str] | None = None,
    ) -> FeatureScore:
        """
        Evaluate feature across all selectors and aggregate votes into a FeatureScore.
        """
        protected_set = set(protected_features or [])
        is_protected = col in protected_set

        votes: dict[str, str] = {}
        reasoning: list[str] = []
        weighted_score_sum = 0.0
        total_weight = 0.0
        has_remove_vote = False

        for selector in selectors:
            weight = rules.selector_weights.get(selector.name, 0.20)
            decision, score, reason = selector.evaluate(col, df, data_profile, rules, target=target)

            dec_str = decision.value if hasattr(decision, "value") else str(decision)
            votes[selector.name] = dec_str

            if dec_str == "remove":
                has_remove_vote = True

            weighted_score_sum += score * weight
            total_weight += weight
            reasoning.append(f"[{selector.name}] {reason}")

        final_score = weighted_score_sum / total_weight if total_weight > 0 else 50.0

        # Protected feature override
        if is_protected:
            return FeatureScore(
                feature_name=col,
                score=100.0,
                confidence=1.0,
                decision=SelectionDecision.KEEP,
                reasoning=["Protected feature override: retained by user request."] + reasoning,
                selector_votes=votes,
                is_protected=True,
            )

        # Consensus decision logic
        rule_vote = votes.get("RuleSelector")
        if rule_vote == "remove" or has_remove_vote and final_score < 60.0:
            final_decision = SelectionDecision.REMOVE
        elif final_score >= 60.0:
            final_decision = SelectionDecision.KEEP
        elif final_score >= 40.0:
            final_decision = SelectionDecision.FLAG
        else:
            final_decision = SelectionDecision.REMOVE

        confidence = min(1.0, max(0.60, abs(final_score - 50.0) / 50.0 + 0.50))

        return FeatureScore(
            feature_name=col,
            score=final_score,
            confidence=confidence,
            decision=final_decision,
            reasoning=reasoning,
            selector_votes=votes,
            is_protected=False,
        )
