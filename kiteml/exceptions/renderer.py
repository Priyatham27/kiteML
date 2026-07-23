"""
renderer.py — Rendering engines for KiteML error formats.
"""

import json
from abc import ABC, abstractmethod
from typing import Any

from kiteml.exceptions.templates import RenderModel
from kiteml.exceptions.themes import Theme, get_theme


class BaseRenderer(ABC):
    """Abstract renderer interface for RenderModel targets."""

    @abstractmethod
    def render(self, model: RenderModel, theme: Theme | None = None) -> str:
        """Render a RenderModel into a string output."""


class TerminalRenderer(BaseRenderer):
    """Primary terminal/notebook renderer formatting errors with dividers, icons, and sections."""

    def render(self, model: RenderModel, theme: Theme | None = None) -> str:
        th = theme or get_theme()
        div = th.divider_char * th.width

        lines = [
            div,
            f"{model.icon} {model.title}" if th.show_icons else model.title,
            div,
        ]

        if th.show_codes and model.error_code:
            lines.extend(["Error Code", model.error_code, div])

        if model.message:
            lines.extend([model.message, div])

        if model.suggestion:
            lines.extend(["Suggestion", model.suggestion, div])

        if model.context_sections:
            lines.append("Context")
            for key, val in model.context_sections.items():
                label = key.replace("_", " ").title()
                if isinstance(val, list):
                    lines.append(f"  {label}")
                    for item in val:
                        lines.append(f"    {th.bullet_char} {item}")
                elif isinstance(val, dict):
                    lines.append(f"  {label}")
                    for sub_k, sub_v in val.items():
                        lines.append(f"    • {sub_k}: {sub_v}")
                else:
                    lines.append(f"  {label:<20} {val}")
            lines.append(div)

        if model.details:
            lines.extend(["Details", model.details, div])

        if th.show_help_url and model.help_url:
            lines.extend([f"📚 Documentation: {model.help_url}", div])

        return "\n".join(lines)


class TextRenderer(BaseRenderer):
    """Plain text renderer optimized for logging."""

    def render(self, model: RenderModel, theme: Theme | None = None) -> str:
        code_part = f"[{model.error_code}] " if model.error_code else ""
        text = f"{code_part}{model.message}"

        if model.suggestion:
            text += f"\nSuggestion: {model.suggestion}"

        if model.context_sections:
            text += "\nContext:"
            for k, v in model.context_sections.items():
                text += f"\n  {k}: {v}"

        return text


class JsonRenderer(BaseRenderer):
    """JSON payload renderer for APIs, web dashboards, and logging."""

    def __init__(self, indent: int = 2) -> None:
        self.indent = indent

    def render(self, model: RenderModel, theme: Theme | None = None) -> str:
        data: dict[str, Any] = {
            "success": False,
            "error_code": model.error_code,
            "message": model.message,
            "severity": model.severity.upper(),
        }
        if model.suggestion:
            data["suggestion"] = model.suggestion
        if model.context_sections:
            data["context"] = model.context_sections
        if model.details:
            data["details"] = model.details
        if model.help_url:
            data["help_url"] = model.help_url

        return json.dumps(data, indent=self.indent)


class HtmlRenderer(BaseRenderer):
    """HTML container renderer for web reports and Jupyter notebooks."""

    def render(self, model: RenderModel, theme: Theme | None = None) -> str:
        html_lines = [
            '<div class="kiteml-error-container" style="border: 2px solid #e53e3e; padding: 16px; border-radius: 8px; font-family: sans-serif;">',
            f'  <h3 style="color: #c53030; margin-top: 0;">{model.icon} {model.title} [{model.error_code}]</h3>',
            f"  <p><strong>{model.message}</strong></p>",
        ]
        if model.suggestion:
            html_lines.append(f'  <p style="color: #2b6cb0;">💡 <strong>Suggestion:</strong> {model.suggestion}</p>')

        if model.context_sections:
            html_lines.append(
                "  <div><strong>Context:</strong><pre>" + json.dumps(model.context_sections, indent=2) + "</pre></div>"
            )

        html_lines.append("</div>")
        return "\n".join(html_lines)


class MarkdownRenderer(BaseRenderer):
    """GitHub-flavored markdown alert block renderer."""

    def render(self, model: RenderModel, theme: Theme | None = None) -> str:
        lines = [
            "> [!CAUTION]",
            f"> **{model.title} [{model.error_code}]**",
            "> ",
            f"> {model.message}",
        ]
        if model.suggestion:
            lines.append("> ")
            lines.append(f"> 💡 **Suggestion:** {model.suggestion}")

        if model.context_sections:
            lines.append("> ")
            lines.append(
                "> ```json\n> " + json.dumps(model.context_sections, indent=2).replace("\n", "\n> ") + "\n> ```"
            )

        return "\n".join(lines)
