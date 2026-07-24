"""
test_hooks.py — Unit tests for HookRegistry (Story 4.7).
"""

import pytest

from kiteml.orchestration import HookRegistry


def test_hook_registry_callbacks():
    registry = HookRegistry()
    called = []

    def callback(ctx):
        called.append("before_validation_triggered")

    registry.register_hook("before_validation", callback)
    registry.trigger("before_validation", context=None)

    assert called == ["before_validation_triggered"]
