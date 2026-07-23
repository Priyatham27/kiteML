"""
hooks.py — Validation pipeline lifecycle hooks for KiteML.
"""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

VALIDATION_EVENTS = {
    "before_validation",
    "after_validator",
    "after_validation",
    "on_validation_failure",
}


class ValidationHookSystem:
    """
    Lifecycle hook system allowing custom extensions and logging callbacks.
    """

    def __init__(self) -> None:
        self._hooks: dict[str, list[Callable[..., Any]]] = {event: [] for event in VALIDATION_EVENTS}

    def register_hook(self, event: str, callback: Callable[..., Any]) -> None:
        """
        Register a callback for a specific validation lifecycle event.

        Parameters
        ----------
        event : str
            One of 'before_validation', 'after_validator', 'after_validation', 'on_validation_failure'.
        callback : Callable
            Function to invoke when event is triggered.
        """
        if event not in self._hooks:
            raise ValueError(f"Unknown validation event '{event}'. Valid events: {VALIDATION_EVENTS}")
        if callback not in self._hooks[event]:
            self._hooks[event].append(callback)

    def trigger_hook(self, event: str, **kwargs: Any) -> None:
        """Trigger all registered callbacks for an event."""
        if event not in self._hooks:
            return

        for callback in self._hooks[event]:
            try:
                callback(**kwargs)
            except Exception as exc:
                logger.warning("⚠️ Exception in validation hook '%s' (%s): %s", event, callback, exc)
