"""
base.py — KiteMLWarning base class inheriting from UserWarning.
"""

import json
import time
from typing import Any

from kiteml.warnings.severity import WarningSeverity, get_warning_icon


class KiteMLWarning(UserWarning):
    """
    Base warning class for all non-fatal KiteML warnings.

    Attributes
    ----------
    message : str
        Main warning message.
    code : str
        Warning code (e.g. 'KML-W-D001').
    severity : WarningSeverity or str
        Severity level ('INFO', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL').
    recommendation : str, optional
        Actionable recommendation to address the issue.
    context : dict, optional
        Key-value contextual data.
    category : str
        Warning category ('Dataset', 'Schema', etc.).
    source : str, optional
        Module or validator source name.
    timestamp : float
        Creation timestamp.
    """

    def __init__(
        self,
        message: str,
        code: str = "KML-W-000",
        severity: WarningSeverity | str = WarningSeverity.MEDIUM,
        recommendation: str | None = None,
        context: dict[str, Any] | None = None,
        category: str = "General",
        source: str | None = None,
        timestamp: float | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.severity = (
            WarningSeverity(severity)
            if isinstance(severity, str) and severity.upper() in WarningSeverity.__members__
            else severity
        )
        self.recommendation = recommendation
        self.context = context or {}
        self.category = category
        self.source = source
        self.timestamp = timestamp if timestamp is not None else time.time()

    def to_dict(self) -> dict[str, Any]:
        """Serialize warning object to a dictionary."""
        result: dict[str, Any] = {
            "warning_class": self.__class__.__name__,
            "code": self.code,
            "category": self.category,
            "severity": self.severity.value if isinstance(self.severity, WarningSeverity) else str(self.severity),
            "message": self.message,
            "timestamp": self.timestamp,
        }
        if self.recommendation:
            result["recommendation"] = self.recommendation
        if self.context:
            result["context"] = self.context
        if self.source:
            result["source"] = self.source
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize warning to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def __str__(self) -> str:
        icon = get_warning_icon(self.severity)
        sev_str = self.severity.value if isinstance(self.severity, WarningSeverity) else str(self.severity)
        output = f"{icon} [{self.code}] [{sev_str}] {self.message}"
        if self.recommendation:
            output += f"\n   → Recommendation: {self.recommendation}"
        return output

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} [{self.code}]: {self.message}>"
