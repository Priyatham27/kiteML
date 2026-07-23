"""
templates.py — RenderModel intermediate representation for KiteML error presentation.
"""

from dataclasses import dataclass, field
from typing import Any

from kiteml.exceptions.base import KiteMLError
from kiteml.exceptions.styles import get_severity_icon


@dataclass
class RenderModel:
    """
    Intermediate representation decoupling exception logic from layout/rendering logic.

    Attributes
    ----------
    title : str
        Display title for the error.
    icon : str
        Status icon string.
    severity : str
        Severity string ('error', 'warning', 'info').
    error_code : str
        Unique error code string (e.g., 'KML-T001').
    message : str
        Main human-readable error description.
    suggestion : str, optional
        Actionable suggestion or fix.
    context_sections : dict, optional
        Key-value or list mapping of structured error context.
    details : str, optional
        Technical stack trace or extra details.
    help_url : str, optional
        Documentation link URL.
    footer : str, optional
        Footer text.
    metadata : dict, optional
        Extra metadata.
    """

    title: str = "KiteML Error"
    icon: str = "❌"
    severity: str = "error"
    error_code: str = "KML-E000"
    message: str = ""
    suggestion: str | None = None
    context_sections: dict[str, Any] = field(default_factory=dict)
    details: str | None = None
    help_url: str | None = None
    footer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def build_render_model(error: KiteMLError) -> RenderModel:
    """
    Construct a RenderModel from a KiteMLError object.

    Parameters
    ----------
    error : KiteMLError
        Exception to convert.

    Returns
    -------
    RenderModel
    """
    severity = getattr(error, "severity", "error").lower()
    icon = get_severity_icon(severity)

    # Context conversion
    context_dict = {}
    if hasattr(error, "context") and error.context:
        context_dict = error.context.to_dict()

    return RenderModel(
        title="KiteML Error",
        icon=icon,
        severity=severity,
        error_code=getattr(error, "error_code", "KML-E000"),
        message=getattr(error, "message", str(error)),
        suggestion=getattr(error, "suggestion", None),
        context_sections=context_dict,
        details=getattr(error, "details", None),
        help_url=getattr(error, "help_url", None),
        metadata={"error_class": type(error).__name__},
    )
