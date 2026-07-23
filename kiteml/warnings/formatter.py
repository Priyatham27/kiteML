"""
formatter.py — Multi-format WarningFormatter for KiteML warnings.
"""

import json
from typing import Any

from kiteml.warnings.base import KiteMLWarning
from kiteml.warnings.report import WarningReport


class WarningFormatter:
    """Formatter for rendering warnings and warning reports in Terminal, Text, or JSON formats."""

    def format(
        self,
        target: KiteMLWarning | WarningReport,
        mode: str = "terminal",
    ) -> str:
        """Format a warning or WarningReport into specified output mode."""
        mode_clean = mode.lower().strip()
        if mode_clean in ("json",):
            return self.to_json(target)
        elif mode_clean in ("text", "plain"):
            return self.to_text(target)
        else:
            return self.to_terminal(target)

    def to_terminal(self, target: KiteMLWarning | WarningReport) -> str:
        """Render target to terminal string format."""
        if isinstance(target, WarningReport):
            return target.summary_text()
        return str(target)

    def to_text(self, target: KiteMLWarning | WarningReport) -> str:
        """Render target to plain text format."""
        if isinstance(target, WarningReport):
            return target.summary_text(width=40)
        return f"[{target.code}] {target.message} (Recommendation: {target.recommendation})"

    def to_json(self, target: KiteMLWarning | WarningReport, indent: int = 2) -> str:
        """Render target to JSON string format."""
        if isinstance(target, WarningReport):
            return json.dumps(target.to_dict(), indent=indent)
        return target.to_json(indent=indent)
