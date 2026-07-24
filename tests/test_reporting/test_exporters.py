"""
test_exporters.py — Unit tests for JSONExporter and HTMLExporter (Story 4.6).
"""

from pathlib import Path

import pytest

from kiteml.reporting import HTMLExporter, JSONExporter, PipelineReport, ReplayEvent


def test_json_and_html_exporters(tmp_path: Path):
    report = PipelineReport()
    report.timeline.add_event(ReplayEvent(stage_name="MissingValueStage", duration_ms=1.2))

    json_str = JSONExporter().export(report, filepath=tmp_path / "report.json")
    assert "MissingValueStage" in json_str
    assert (tmp_path / "report.json").exists()

    html_str = HTMLExporter().export(report, filepath=tmp_path / "report.html")
    assert "KiteML Transformation Pipeline Report" in html_str
    assert "MissingValueStage" in html_str
    assert (tmp_path / "report.html").exists()
