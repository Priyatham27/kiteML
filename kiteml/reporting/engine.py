"""
engine.py — Public pipeline report generation functions for KiteML.
"""

from typing import Any

from kiteml.reporting.builder import ReportBuilder
from kiteml.reporting.report import PipelineReport


def generate_report(pipeline: Any) -> PipelineReport:
    """
    Generate a comprehensive PipelineReport for a fitted TransformationPipeline.

    Parameters
    ----------
    pipeline : TransformationPipeline
        Fitted transformation pipeline.

    Returns
    -------
    PipelineReport
        Pipeline execution report.
    """
    builder = ReportBuilder()
    return builder.build_report(pipeline)
