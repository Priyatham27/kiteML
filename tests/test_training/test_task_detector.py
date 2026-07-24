"""
test_task_detector.py — Unit tests for TaskDetector (Story 5.1).
"""

import pandas as pd
import pytest

from kiteml.training import TaskDetector


def test_task_detector_binary():
    s = pd.Series([0, 1, 0, 1, 1])
    assert TaskDetector().detect_task(s) == "binary_classification"


def test_task_detector_regression():
    s = pd.Series([10.5, 12.3, 14.1, 19.8, 22.0, 31.2])
    assert TaskDetector().detect_task(s) == "regression"


def test_task_detector_multiclass():
    s = pd.Series(["cat", "dog", "bird", "cat", "dog"])
    assert TaskDetector().detect_task(s) == "multiclass_classification"
