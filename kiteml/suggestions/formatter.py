"""
formatter.py — Multi-format SuggestionFormatter for KiteML suggestions.
"""

import json
from typing import Sequence

from kiteml.suggestions.result import Suggestion


class SuggestionFormatter:
    """Formatter for rendering Suggestion objects into Terminal, Text, or JSON formats."""

    def format(
        self,
        suggestions: Sequence[Suggestion],
        mode: str = "terminal",
        width: int = 50,
    ) -> str:
        """Format a list of suggestions into specified mode."""
        mode_clean = mode.lower().strip()
        if mode_clean in ("json",):
            return self.to_json(suggestions)
        elif mode_clean in ("text", "plain"):
            return self.to_text(suggestions)
        else:
            return self.to_terminal(suggestions, width=width)

    def to_terminal(self, suggestions: Sequence[Suggestion], width: int = 50) -> str:
        """Render suggestions to terminal string format with explainable rationale."""
        if not suggestions:
            return "💡 No suggestions available."

        lines = [
            "━" * width,
            "💡 KiteML Suggestions & Recommendations",
            "━" * width,
        ]

        for s in suggestions:
            lines.append(f"✓ {s.title} ({s.confidence_percentage}% confidence)")
            lines.append(f"  Category: {s.category}")
            lines.append(f"  {s.description}")
            if s.action:
                lines.append(f"  Action: {s.action}")
            if s.why:
                lines.append("  Why?")
                for reason in s.why:
                    lines.append(f"    • {reason}")
            lines.append("─" * width)

        return "\n".join(lines)

    def to_text(self, suggestions: Sequence[Suggestion]) -> str:
        """Render suggestions to plain text format."""
        if not suggestions:
            return "No suggestions available."
        return "\n\n".join([str(s) for s in suggestions])

    def to_json(self, suggestions: Sequence[Suggestion], indent: int = 2) -> str:
        """Render suggestions to JSON string format."""
        return json.dumps([s.to_dict() for s in suggestions], indent=indent)
