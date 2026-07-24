"""
registry/ — Intelligent Model Registry package for KiteML.
"""

from kiteml.registry.capabilities import CapabilityAnalyzer, CapabilityScore
from kiteml.registry.factory import ModelFactory
from kiteml.registry.model_info import ModelInfo
from kiteml.registry.plugins import PluginLoader
from kiteml.registry.providers import ModelProvider
from kiteml.registry.registry import ModelRegistry, model_registry

__all__ = [
    "ModelRegistry",
    "model_registry",
    "ModelProvider",
    "ModelInfo",
    "ModelFactory",
    "CapabilityAnalyzer",
    "CapabilityScore",
    "PluginLoader",
]
