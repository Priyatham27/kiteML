"""
test_early_stopping.py — Unit tests for EarlyStopping (Story 5.3).
"""

import pytest

from kiteml.optimization import EarlyStopping


def test_early_stopping_max_trials():
    es = EarlyStopping(max_trials=3)
    assert not es.should_stop(trial_count=1, elapsed_time=1.0, current_score=0.8)
    assert not es.should_stop(trial_count=2, elapsed_time=2.0, current_score=0.85)
    assert es.should_stop(trial_count=3, elapsed_time=3.0, current_score=0.86)
