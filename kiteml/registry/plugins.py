"""
plugins.py — PluginLoader for third-party ModelProvider extensions in KiteML registry.
"""

from typing import Any

from kiteml.registry.providers import ModelProvider
from kiteml.registry.registry import ModelRegistry


class PluginLoader:
    """
    Discovers and registers external third-party ModelProvider plugins.
    """

    def register_plugin(self, registry: ModelRegistry, provider: Any) -> None:
        """
        Register a third-party plugin provider.

        Parameters
        ----------
        registry : ModelRegistry
            Target model registry instance.
        provider : Any
            Object implementing ModelProvider interface.
        """
        if not isinstance(provider, ModelProvider):
            raise TypeError("Plugin provider must inherit from ModelProvider.")

        registry.register(provider)
