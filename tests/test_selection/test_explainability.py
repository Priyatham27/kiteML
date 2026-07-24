"""
test_explainability.py — Unit tests for SelectionExplainer (Story 5.5).
"""

import pytest

from kiteml.selection import SelectionExplainer


def test_selection_explainer():
    explainer = SelectionExplainer()
    explanation = explainer.explain_selection(
        winner_name="RandomForest",
        winner_score=92.5,
        runner_up_name="LogisticRegression",
        runner_up_score=85.0,
        policy="balanced",
    )

    assert "RandomForest" in explanation
    assert "LogisticRegression" in explanation
    assert "92.50" in explanation
