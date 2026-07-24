"""
test_result.py — Unit tests for TrainingResult (Story 5.8).
"""

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from kiteml.ml import TrainingResult, UnifiedReport
from kiteml.selection.best_model import BestModel


def test_training_result(tmp_path):
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit([[1], [2]], [0, 1])

    bm = BestModel(model=clf, model_name="Dummy", composite_score=80.0)
    report = UnifiedReport("Dummy", "classification", 80.0, "Baseline")

    res = TrainingResult(
        best_model=bm,
        pipeline=None,
        problem_type="classification",
        report=report,
        metrics={},
    )

    preds = res.predict(pd.DataFrame({"a": [1]}))
    assert len(preds) == 1

    save_report = res.save(tmp_path / "model.kiteml")
    assert save_report.is_valid is True
