"""
test_splitter.py — Unit tests for DataSplitter (Story 5.1).
"""

import pandas as pd
import pytest

from kiteml.training import DataSplitter


def test_data_splitter():
    X = pd.DataFrame({"a": range(10), "b": range(10)})
    y = pd.Series([0, 1] * 5)

    X_tr, X_te, y_tr, y_te = DataSplitter().split(
        X, y, test_size=0.2, random_state=42, task_type="binary_classification"
    )

    assert len(X_tr) == 8
    assert len(X_te) == 2
