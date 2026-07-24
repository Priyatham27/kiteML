"""
test_trials.py — Unit tests for TrialManager (Story 5.3).
"""

import pytest

from kiteml.optimization import OptimizationTrial, TrialManager


def test_trial_manager():
    tm = TrialManager()
    t1 = OptimizationTrial(trial_id=1, parameters={"a": 1}, score=0.8)
    t2 = OptimizationTrial(trial_id=2, parameters={"a": 2}, score=0.9)

    tm.record_trial(t1)
    tm.record_trial(t2)

    best = tm.get_best_trial()
    assert best is not None
    assert best.trial_id == 2
    assert best.score == 0.9
