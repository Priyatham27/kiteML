"""
exporters.py — JSON and HTML report exporters for KiteML pipeline reporting.
"""

import json
from pathlib import Path
from typing import Any


class JSONExporter:
    """Exporter formatting PipelineReport into JSON text or file."""

    def export(self, report: Any, filepath: str | Path | None = None) -> str:
        """Export report to JSON string or file."""
        json_str = json.dumps(report.to_dict(), indent=2)
        if filepath:
            out_path = Path(filepath)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json_str, encoding="utf-8")
        return json_str


class HTMLExporter:
    """Exporter rendering PipelineReport into a styled HTML dashboard report."""

    def export(self, report: Any, filepath: str | Path | None = None) -> str:
        """Render report into HTML dashboard string or file."""
        stats = report.statistics
        timeline_items = ""
        for evt in report.timeline.events:
            timeline_items += f"""
            <div style="padding: 10px; margin-bottom: 8px; border-left: 4px solid #4CAF50; background: #f9f9f9;">
                <strong>{evt.stage_name}</strong> — {evt.duration_ms:.1f} ms<br/>
                <small>Shape: {evt.input_shape} → {evt.output_shape}</small>
            </div>
            """

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>KiteML Pipeline Execution Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; color: #333; background: #fafafa; }}
        .card {{ background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        h1 {{ color: #2C3E50; }}
        .metric-grid {{ display: flex; gap: 20px; margin-bottom: 20px; }}
        .metric-card {{ flex: 1; background: #EDF2F7; padding: 15px; border-radius: 6px; text-align: center; }}
        .metric-val {{ font-size: 24px; font-weight: bold; color: #2B6CB0; }}
    </style>
</head>
<body>
    <h1>🪁 KiteML Transformation Pipeline Report</h1>

    <div class="metric-grid">
        <div class="metric-card">
            <div>Initial Shape</div>
            <div class="metric-val">{stats.initial_rows} × {stats.initial_cols}</div>
        </div>
        <div class="metric-card">
            <div>Final Shape</div>
            <div class="metric-val">{stats.final_rows} × {stats.final_cols}</div>
        </div>
        <div class="metric-card">
            <div>Generated Features</div>
            <div class="metric-val">{stats.generated_features_count}</div>
        </div>
        <div class="metric-card">
            <div>Dropped Features</div>
            <div class="metric-val">{stats.dropped_features_count}</div>
        </div>
        <div class="metric-card">
            <div>Execution Time</div>
            <div class="metric-val">{stats.total_execution_time:.3f} s</div>
        </div>
    </div>

    <div class="card">
        <h2>⏱️ Execution Replay Timeline</h2>
        {timeline_items}
    </div>
</body>
</html>
"""

        if filepath:
            out_path = Path(filepath)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(html_content, encoding="utf-8")

        return html_content
