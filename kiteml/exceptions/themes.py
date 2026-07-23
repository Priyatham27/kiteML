"""
themes.py — Layout theme configurations for KiteML error formatting.
"""

from dataclasses import dataclass
from typing import Any

from kiteml.exceptions.styles import BULLET_POINT, DEFAULT_DIVIDER_CHAR, DEFAULT_WIDTH


@dataclass
class Theme:
    """
    Theme configuration for formatting error displays.

    Attributes
    ----------
    name : str
        Theme identifier.
    divider_char : str
        Character used for section dividers.
    width : int
        Width of terminal lines.
    bullet_char : str
        Character used for list bullets.
    show_icons : bool
        Whether to include status icons.
    show_codes : bool
        Whether to display error codes.
    show_help_url : bool
        Whether to render help URL link.
    """

    name: str = "default"
    divider_char: str = DEFAULT_DIVIDER_CHAR
    width: int = DEFAULT_WIDTH
    bullet_char: str = BULLET_POINT
    show_icons: bool = True
    show_codes: bool = True
    show_help_url: bool = True


DEFAULT_THEME = Theme(name="default")
MINIMAL_THEME = Theme(
    name="minimal",
    divider_char="─",
    show_icons=False,
    show_help_url=False,
)
RICH_THEME = Theme(
    name="rich",
    divider_char="═",
    width=60,
)

THEMES: dict[str, Theme] = {
    "default": DEFAULT_THEME,
    "minimal": MINIMAL_THEME,
    "rich": RICH_THEME,
}


def get_theme(name: str | None = None) -> Theme:
    """Retrieve theme by name, fallback to DEFAULT_THEME."""
    if not name:
        return DEFAULT_THEME
    return THEMES.get(name.lower(), DEFAULT_THEME)
