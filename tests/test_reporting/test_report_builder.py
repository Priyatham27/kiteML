"""
test_report_builder.py — Unit tests for ReportBuilder (Story 4.6).
"""

import pandas as pd
import pytest

from kiteml.pipeline import TransformationPipeline
from kiteml.reporting import ReportBuilder


def test_report_builder():
    df = pd.DataFrame({"price": [10.0, 20.0], "target": [0, 1]})
    pipeline = TransformationPipeline()
    pipeline.fit(df, target="target")

    report = ReportBuilder().build_report(pipeline)

    assert report.statistics.initial_rows == 2
    assert "KiteML Transformation Pipeline Report" in report.summary()
