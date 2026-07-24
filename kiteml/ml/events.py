"""
events.py — MLWorkflowEventBus for event broadcasting across ML workflow DAG in KiteML.
"""

import contextlib
from typing import Any, Callable


class MLWorkflowEventBus:
    """
    Event bus broadcasting stage transitions across the ML training workflow DAG.
    """

    def __init__(self) -> None:
        self.subscribers: dict[str, list[Callable[[dict[str, Any]], None]]] = {}

    def subscribe(self, event_type: str, callback: Callable[[dict[str, Any]], None]) -> None:
        """Subscribe callback to event type."""
        self.subscribers.setdefault(event_type, []).append(callback)

    def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        """Publish event payload to subscribers."""
        for cb in self.subscribers.get(event_type, []):
            with contextlib.suppress(Exception):
                cb(payload)
