"""
test_packager.py — Unit tests for PackageBuilder (Story 5.7).
"""

import pytest
from sklearn.dummy import DummyClassifier

from kiteml.deployment import PackageBuilder


def test_package_builder(tmp_path):
    clf = DummyClassifier()
    clf.fit([[1]], [0])

    builder = PackageBuilder()
    out = tmp_path / "model.kiteml"

    pkg_file = builder.build_package(
        model=clf,
        model_name="DummyClassifier",
        task_type="classification",
        output_path=out,
    )

    assert pkg_file.exists()
    assert pkg_file.name.endswith(".kiteml")
