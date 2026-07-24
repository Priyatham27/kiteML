"""
test_voting_system.py — Unit tests for FeatureSelectionVotingSystem (Story 4.3).
"""

import pandas as pd
import pytest

from kiteml.feature_selection import (
    FeatureSelectionVotingSystem,
    FSRuleEngine,
    RuleSelector,
    SelectionDecision,
    VarianceSelector,
)


def test_voting_system_consensus():
    df = pd.DataFrame({"const": [5] * 20})
    voting = FeatureSelectionVotingSystem()
    selectors = [RuleSelector(), VarianceSelector()]
    rules = FSRuleEngine()

    score_obj = voting.evaluate_feature("const", df, selectors, rules)

    assert score_obj.decision == SelectionDecision.REMOVE
    assert score_obj.score < 50.0
    assert "RuleSelector" in score_obj.selector_votes
    assert score_obj.selector_votes["RuleSelector"] == "remove"


def test_voting_system_protected_override():
    df = pd.DataFrame({"const": [5] * 20})
    voting = FeatureSelectionVotingSystem()
    selectors = [RuleSelector(), VarianceSelector()]
    rules = FSRuleEngine()

    score_obj = voting.evaluate_feature(
        "const",
        df,
        selectors,
        rules,
        protected_features=["const"],
    )

    assert score_obj.decision == SelectionDecision.KEEP
    assert score_obj.is_protected is True
    assert score_obj.score == 100.0
