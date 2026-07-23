"""
styles.py — Standardized icons, dividers, and spacing for KiteML error formatting.
"""

# Icons
ERROR_ICON = "❌"
WARNING_ICON = "⚠"
INFO_ICON = "ℹ"
SUCCESS_ICON = "✅"

SEVERITY_ICONS = {
    "error": ERROR_ICON,
    "critical": ERROR_ICON,
    "warning": WARNING_ICON,
    "info": INFO_ICON,
    "success": SUCCESS_ICON,
}

# Dividers
DEFAULT_DIVIDER_CHAR = "━"
DEFAULT_WIDTH = 50

HEAVY_DIVIDER = DEFAULT_DIVIDER_CHAR * DEFAULT_WIDTH
LIGHT_DIVIDER = "─" * DEFAULT_WIDTH
BULLET_POINT = "•"


def get_severity_icon(severity: str) -> str:
    """Return standardized icon for a given severity string."""
    return SEVERITY_ICONS.get(severity.lower(), ERROR_ICON)
