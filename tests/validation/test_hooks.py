"""
test_hooks.py — Unit tests for ValidationHookSystem (Story 2.6).
"""

import pandas as pd
import pytest

from kiteml.validation.hooks import ValidationHookSystem
from kiteml.validation.pipeline import ValidationPipeline


def test_hook_registration_and_execution():
    hooks = ValidationHookSystem()
    events_triggered = []

    def before_cb(**kwargs):
        events_triggered.append("before_validation")

    def after_val_cb(**kwargs):
        events_triggered.append("after_validator")

    def after_cb(**kwargs):
        events_triggered.append("after_validation")

    hooks.register_hook("before_validation", before_cb)
    hooks.register_hook("after_validator", after_val_cb)
    hooks.register_hook("after_validation", after_cb)

    pipeline = ValidationPipeline(hooks=hooks)
    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    pipeline.validate(df, target="target")

    assert "before_validation" in events_triggered
    assert "after_validator" in events_triggered
    assert "after_validation" in events_triggered


def test_failure_hook():
    hooks = ValidationHookSystem()
    failures = []

    def on_fail_cb(**kwargs):
        failures.append(kwargs.get("validator_name"))

    hooks.register_hook("on_validation_failure", on_fail_cb)

    pipeline = ValidationPipeline(hooks=hooks)
    pipeline.validate(None, target="target", fail_fast=True)

    assert "DatasetValidator" in failures
