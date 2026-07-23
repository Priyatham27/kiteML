"""
severity.py — Severity levels and icons for KiteML warnings.
"""

from enum import Enum


class WarningSeverity(str, Enum):
    """Warning severity levels."""

    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


WARNING_SEVERITY_ICONS = {
    WarningSeverity.INFO: "ℹ",
    WarningSeverity.LOW: "⚠",
    WarningSeverity.MEDIUM: "⚠",
    WarningSeverity.HIGH: "⚠",
    WarningSeverity.CRITICAL: "🔴",
}


def get_warning_icon(severity: WarningSeverity | str) -> str:
    """Return icon for warning severity level."""
    sev_str = severity.value if isinstance(severity, WarningSeverity) else str(severity).upper()
    try:
        sev_enum = WarningSeverity(sev_str)
        return WARNING_SEVERITY_ICONS.get(sev_enum, "⚠")
    except ValueError:
        return "⚠"
