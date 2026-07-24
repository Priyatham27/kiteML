"""
test_events_hooks.py — Unit tests for MLWorkflowEventBus and MLHookRegistry (Story 5.8).
"""

import pytest

from kiteml.ml import MLHookRegistry, MLWorkflowEventBus


def test_events_and_hooks():
    bus = MLWorkflowEventBus()
    received = []

    bus.subscribe("TestEvent", lambda p: received.append(p["val"]))
    bus.publish("TestEvent", {"val": 42})

    assert received == [42]

    hooks = MLHookRegistry()
    pre_called = []
    hooks.register_pre_hook("analyze", lambda ctx: pre_called.append(True))
    hooks.run_pre_hooks("analyze", None)

    assert pre_called == [True]
