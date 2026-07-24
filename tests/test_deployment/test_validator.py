"""
test_validator.py — Unit tests for PackageValidator (Story 5.7).
"""

import pytest
from sklearn.dummy import DummyClassifier

from kiteml.deployment import PackageBuilder, PackageValidator


def test_package_validator(tmp_path):
    clf = DummyClassifier()
    clf.fit([[1]], [0])

    pkg_path = PackageBuilder().build_package(
        model=clf,
        model_name="Dummy",
        task_type="classification",
        output_path=tmp_path / "valid.kiteml",
    )

    validator = PackageValidator()
    assert validator.validate_package(pkg_path) is True
