"""
test_loader.py — Unit tests for PackageLoader (Story 5.7).
"""

import pytest
from sklearn.dummy import DummyClassifier

from kiteml.deployment import LoadedPackage, PackageBuilder, PackageLoader


def test_package_loader(tmp_path):
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit([[1], [2]], [0, 1])

    pkg_path = PackageBuilder().build_package(
        model=clf,
        model_name="DummyClassifier",
        task_type="classification",
        output_path=tmp_path / "dummy.kiteml",
    )

    loaded = PackageLoader().load_package(pkg_path)
    assert isinstance(loaded, LoadedPackage)
    assert loaded.manifest["model_name"] == "DummyClassifier"

    preds = loaded.predict([[1]])
    assert len(preds) == 1
