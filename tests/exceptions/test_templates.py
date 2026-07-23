"""
test_templates.py — Unit tests for RenderModel and build_render_model (Story 3.2).
"""

import pytest

from kiteml.exceptions import KiteMLError, TargetError
from kiteml.exceptions.templates import RenderModel, build_render_model


def test_build_render_model_basic():
    err = TargetError(
        message='Target column "price" not found.',
        error_code="KML-T001",
        suggestion="Choose an existing target column.",
        context={"available_columns": ["age", "salary", "city"]},
    )
    model = build_render_model(err)

    assert isinstance(model, RenderModel)
    assert model.title == "KiteML Error"
    assert model.icon == "❌"
    assert model.error_code == "KML-T001"
    assert model.message == 'Target column "price" not found.'
    assert model.suggestion == "Choose an existing target column."
    assert model.context_sections["available_columns"] == ["age", "salary", "city"]


def test_build_render_model_warning_severity():
    err = KiteMLError(message="Low dataset rows", severity="warning")
    model = build_render_model(err)

    assert model.icon == "⚠"
    assert model.severity == "warning"
