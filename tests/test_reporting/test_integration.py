"""
test_integration.py — Integration tests for pipeline.report() (Story 4.6).
"""

import pandas as pd
import pytest

from kiteml.pipeline import TransformationPipeline
from kiteml.reporting import PipelineReport


def test_pipeline_report_method_integration():
    df = pd.DataFrame(
        {
            "price": [10.0, 20.0, 30.0, 40.0],
            "quantity": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )

    pipeline = TransformationPipeline()
    pipeline.fit(df, target="target")

    report = pipeline.report()

    assert isinstance(report, PipelineReport)
    assert len(report.timeline.events) > 0
    assert "MissingValueStage" in [e.stage_name for e in report.timeline.events]
