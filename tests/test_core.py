"""
Tests for kiteml.core module.
"""

import pytest
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes

from kiteml.core import train


class TestTrainClassification:
    """Tests for classification tasks."""

    def test_train_iris_auto_detect(self):
        """Train on Iris dataset with auto problem type detection."""
        iris = load_iris(as_frame=True)
        df = iris.frame
        result = train(df, target="target", verbose=False)

        assert result.problem_type == "classification"
        assert result.metrics.accuracy > 0.5
        assert result.model is not None

    def test_train_with_explicit_type(self):
        """Train with explicit problem type."""
        iris = load_iris(as_frame=True)
        df = iris.frame
        result = train(df, target="target", problem_type="classification", verbose=False)

        assert result.problem_type == "classification"


class TestTrainRegression:
    """Tests for regression tasks."""

    def test_train_diabetes(self):
        """Train on Diabetes dataset for regression."""
        diabetes = load_diabetes(as_frame=True)
        df = diabetes.frame
        result = train(df, target="target", verbose=False)

        assert result.problem_type == "regression"
        assert hasattr(result.metrics, "r2_score")
        assert result.model is not None


class TestTrainFromCSV:
    """Tests for loading from file paths."""

    def test_train_from_csv(self, tmp_path):
        """Train from a CSV file."""
        iris = load_iris(as_frame=True)
        df = iris.frame
        csv_path = tmp_path / "iris.csv"
        df.to_csv(csv_path, index=False)

        result = train(str(csv_path), target="target", verbose=False)
        assert result.model is not None
