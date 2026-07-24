"""
test_plugins.py — Unit tests for PluginLoader (Story 5.2).
"""

from typing import Any

import pytest

from kiteml.registry import ModelInfo, ModelProvider, ModelRegistry, PluginLoader


class CustomDummyProvider(ModelProvider):

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="CustomDummy",
            family="custom",
            task_types=["classification"],
        )

    def create(self, params: dict[str, Any] | None = None) -> Any:
        return "CustomDummyInstance"


def test_plugin_loader():
    reg = ModelRegistry()
    loader = PluginLoader()
    custom_prov = CustomDummyProvider()

    loader.register_plugin(reg, custom_prov)
    assert "CustomDummy" in reg.list_models()
    assert reg.create("CustomDummy") == "CustomDummyInstance"
