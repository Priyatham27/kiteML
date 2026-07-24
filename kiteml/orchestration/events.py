"""
events.py — OrchestrationEventBus for event notifications during KiteML pipeline execution.
"""

import contextlib
import datetime
from typing import Any, Callable


class OrchestrationEventBus:
    """
    Event bus broadcasting pipeline execution events to subscribers and maintaining an event log.
    """

    def __init__(self) -> None:
        self.subscribers: dict[str, list[Callable[[dict[str, Any]], None]]] = {}
        self.event_log: list[dict[str, Any]] = []

    def subscribe(self, event_name: str, handler: Callable[[dict[str, Any]], None]) -> None:
        """Subscribe a handler function to an event."""
        if event_name not in self.subscribers:
            self.subscribers[event_name] = []
        self.subscribers[event_name].append(handler)

    def emit(self, event_name: str, payload: dict[str, Any] | None = None) -> None:
        """Emit an event to all subscribers and append to event log."""
        event_entry = {
            "event_name": event_name,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "payload": payload or {},
        }
        self.event_log.append(event_entry)

        if event_name in self.subscribers:
            for handler in self.subscribers[event_name]:
                with contextlib.suppress(Exception):
                    handler(event_entry)
