"""
test_adapters.py — Unit tests for FastAPIAdapter, JoblibAdapter, and PickleAdapter (Story 5.7).
"""

import pytest
from sklearn.dummy import DummyClassifier

from kiteml.deployment import FastAPIAdapter, JoblibAdapter, PickleAdapter


def test_fastapi_adapter(tmp_path):
    clf = DummyClassifier()
    clf.fit([[1]], [0])

    out = FastAPIAdapter().export(model=clf, output_dir=tmp_path / "api_export")
    assert (out / "app.py").exists()
    assert (out / "model.pkl").exists()


def test_joblib_adapter(tmp_path):
    clf = DummyClassifier()
    clf.fit([[1]], [0])

    out = JoblibAdapter().export(model=clf, output_dir=tmp_path / "model.joblib")
    assert out.exists()


def test_pickle_adapter(tmp_path):
    clf = DummyClassifier()
    clf.fit([[1]], [0])

    out = PickleAdapter().export(model=clf, output_dir=tmp_path / "model.pkl")
    assert out.exists()
