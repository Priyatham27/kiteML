"""
formatter.py — Centralized ErrorFormatter orchestrator for KiteML.
"""

from typing import Any

from kiteml.exceptions.base import KiteMLError
from kiteml.exceptions.renderer import (
    HtmlRenderer,
    JsonRenderer,
    MarkdownRenderer,
    TerminalRenderer,
    TextRenderer,
)
from kiteml.exceptions.templates import build_render_model
from kiteml.exceptions.themes import Theme, get_theme


class ErrorFormatter:
    """
    Centralized Error Formatter converting KiteMLError objects into formatted outputs.
    """

    def __init__(self, theme: Theme | str | None = None) -> None:
        self.theme = get_theme(theme) if isinstance(theme, str) else theme or get_theme()
        self._terminal_renderer = TerminalRenderer()
        self._text_renderer = TextRenderer()
        self._html_renderer = HtmlRenderer()
        self._markdown_renderer = MarkdownRenderer()

    def format(
        self,
        error: KiteMLError,
        mode: str = "terminal",
        theme: Theme | str | None = None,
    ) -> str:
        """
        Format a KiteMLError into the specified mode string.

        Parameters
        ----------
        error : KiteMLError
            Exception object.
        mode : str
            'terminal', 'text', 'json', 'html', or 'markdown'.
        theme : Theme or str, optional
            Theme override.

        Returns
        -------
        str
        """
        th = get_theme(theme) if isinstance(theme, str) else theme or self.theme
        model = build_render_model(error)

        mode_clean = mode.lower().strip()
        if mode_clean == "terminal":
            return self._terminal_renderer.render(model, theme=th)
        elif mode_clean in ("text", "plain"):
            return self._text_renderer.render(model, theme=th)
        elif mode_clean == "json":
            return JsonRenderer().render(model, theme=th)
        elif mode_clean == "html":
            return self._html_renderer.render(model, theme=th)
        elif mode_clean in ("markdown", "md"):
            return self._markdown_renderer.render(model, theme=th)
        else:
            return self._terminal_renderer.render(model, theme=th)

    def to_terminal(self, error: KiteMLError, theme: Theme | str | None = None) -> str:
        """Format error as terminal display string."""
        return self.format(error, mode="terminal", theme=theme)

    def to_text(self, error: KiteMLError) -> str:
        """Format error as plain text string for logging."""
        return self.format(error, mode="text")

    def to_json(self, error: KiteMLError, indent: int = 2) -> str:
        """Format error as JSON payload string."""
        model = build_render_model(error)
        return JsonRenderer(indent=indent).render(model)

    def to_dict(self, error: KiteMLError) -> dict[str, Any]:
        """Convert error to dictionary."""
        return error.to_dict()

    def to_html(self, error: KiteMLError) -> str:
        """Format error as HTML container string."""
        return self.format(error, mode="html")

    def to_markdown(self, error: KiteMLError) -> str:
        """Format error as GitHub-flavored Markdown alert string."""
        return self.format(error, mode="markdown")
