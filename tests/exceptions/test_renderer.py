"""
test_renderer.py — Unit tests for error renderers (Story 3.2).
"""

import json

import pytest

from kiteml.exceptions.renderer import (
    HtmlRenderer,
    JsonRenderer,
    MarkdownRenderer,
    TerminalRenderer,
    TextRenderer,
)
from kiteml.exceptions.templates import RenderModel


@pytest.fixture
def sample_model():
    return RenderModel(
        title="KiteML Error",
        icon="❌",
        severity="error",
        error_code="KML-D004",
        message="Dataset contains zero rows.",
        suggestion="Verify dataset before training.",
        context_sections={"rows": 0, "columns": 18, "available_columns": ["col1", "col2"]},
    )


def test_terminal_renderer(sample_model):
    renderer = TerminalRenderer()
    output = renderer.render(sample_model)

    assert "❌ KiteML Error" in output
    assert "KML-D004" in output
    assert "Dataset contains zero rows." in output
    assert "Verify dataset before training." in output
    assert "Available Columns" in output
    assert "col1" in output


def test_text_renderer(sample_model):
    renderer = TextRenderer()
    output = renderer.render(sample_model)

    assert "[KML-D004] Dataset contains zero rows." in output
    assert "Suggestion: Verify dataset before training." in output
    assert "rows: 0" in output


def test_json_renderer(sample_model):
    renderer = JsonRenderer()
    output = renderer.render(sample_model)

    parsed = json.loads(output)
    assert parsed["success"] is False
    assert parsed["error_code"] == "KML-D004"
    assert parsed["message"] == "Dataset contains zero rows."
    assert parsed["context"]["rows"] == 0


def test_html_renderer(sample_model):
    renderer = HtmlRenderer()
    output = renderer.render(sample_model)

    assert '<div class="kiteml-error-container"' in output
    assert "KML-D004" in output
    assert "Dataset contains zero rows." in output


def test_markdown_renderer(sample_model):
    renderer = MarkdownRenderer()
    output = renderer.render(sample_model)

    assert "> [!CAUTION]" in output
    assert "KML-D004" in output
    assert "Dataset contains zero rows." in output
