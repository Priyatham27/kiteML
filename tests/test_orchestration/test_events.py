"""
test_events.py — Unit tests for OrchestrationEventBus (Story 4.7).
"""

import pytest

from kiteml.orchestration import OrchestrationEventBus


def test_event_bus_emission():
    bus = OrchestrationEventBus()
    received = []

    def handler(event_data):
        received.append(event_data["event_name"])

    bus.subscribe("DatasetValidated", handler)
    bus.emit("DatasetValidated", {"rows": 100})

    assert received == ["DatasetValidated"]
    assert len(bus.event_log) == 1
