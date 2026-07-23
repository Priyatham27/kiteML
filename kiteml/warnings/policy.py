"""
policy.py — WarningPolicy engine for configurable warning behavior in KiteML.
"""

from dataclasses import dataclass, field
from typing import Any

from kiteml.exceptions.base import KiteMLError
from kiteml.warnings.base import KiteMLWarning


@dataclass
class WarningPolicy:
    """
    Policy engine controlling how KiteML handles warnings.

    Actions:
    - 'ignore' : Drop the warning silently.
    - 'info'   : Downgrade to info log.
    - 'warn'   : Standard behavior (record in WarningReport).
    - 'error'  : Escalate to a KiteMLError exception.

    Parameters
    ----------
    default_action : str
        Default policy action ('warn', 'ignore', 'info', 'error').
    code_actions : dict
        Map of warning code -> action ('KML-W-D001': 'error').
    category_actions : dict
        Map of category -> action ('Schema': 'ignore').
    """

    default_action: str = "warn"
    code_actions: dict[str, str] = field(default_factory=dict)
    category_actions: dict[str, str] = field(default_factory=dict)

    def get_action(self, warning: KiteMLWarning) -> str:
        """Determine action for a given warning."""
        if warning.code in self.code_actions:
            return self.code_actions[warning.code].lower()
        if warning.category in self.category_actions:
            return self.category_actions[warning.category].lower()
        return self.default_action.lower()

    def process(self, warning: KiteMLWarning) -> KiteMLWarning | None:
        """
        Process warning according to policy.

        Returns warning if action is 'warn' or 'info', returns None if 'ignore',
        and raises KiteMLError if action is 'error'.
        """
        action = self.get_action(warning)
        if action == "ignore":
            return None
        elif action == "error":
            raise KiteMLError(
                message=f"[Escalated Warning {warning.code}] {warning.message}",
                error_code=warning.code.replace("KML-W-", "KML-"),
                suggestion=warning.recommendation,
                context=warning.context,
            )
        return warning
