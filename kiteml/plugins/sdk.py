"""
plugins/sdk.py — SDK for authoring KiteML Plugins.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class KiteMLPlugin(ABC):
    """Base class for all KiteML ecosystem plugins."""

    name: str = "base_plugin"
    version: str = "0.1.0"

    @abstractmethod
    def initialize(self) -> None:
        """Called when the plugin is loaded into the KiteML registry."""
        pass


class ModelPlugin(KiteMLPlugin):
    """Plugin to register new custom models to the training engine."""

    @abstractmethod
    def get_models(self) -> Dict[str, Any]:
        """Return a mapping of model names to uninstantiated sklearn-compatible classes."""
        pass


class ExporterPlugin(KiteMLPlugin):
    """Plugin to add custom export formats (e.g. TensorRT, CoreML)."""

    @abstractmethod
    def export(self, bundle_path: str, output_path: str) -> None:
        """Convert a KiteML bundle into the target format."""
        pass
