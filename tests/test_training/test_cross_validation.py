"""
test_cross_validation.py — Unit tests for CrossValidationEngine (Story 5.1).
"""

from sklearn.model_selection import KFold, StratifiedKFold

from kiteml.training import CrossValidationEngine


def test_cross_validation_engine():
    cve = CrossValidationEngine()

    strat_cv = cve.get_cv("binary_classification", n_splits=5)
    assert isinstance(strat_cv, StratifiedKFold)

    kf_cv = cve.get_cv("regression", n_splits=5)
    assert isinstance(kf_cv, KFold)
