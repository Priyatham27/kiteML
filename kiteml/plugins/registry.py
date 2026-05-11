"""
plugins/registry.py — Plugin registry and loader.
"""

import importlib
import pkgutil
from typing import Dict, List

from kiteml.plugins.sdk import KiteMLPlugin

_PLUGINS: Dict[str, KiteMLPlugin] = {}


def discover_plugins() -> List[str]:
    """Discover plugins installed via pip using the 'kiteml_' prefix namespace."""
    discovered = []
    for finder, name, ispkg in pkgutil.iter_modules():
        if name.startswith("kiteml_"):
            discovered.append(name)
    return discovered


def load_plugin(module_name: str) -> bool:
    """Load and initialize a specific plugin module."""
    try:
        mod = importlib.import_module(module_name)
        if hasattr(mod, "get_plugin"):
            plugin_instance = mod.get_plugin()
            plugin_instance.initialize()
            _PLUGINS[plugin_instance.name] = plugin_instance
            return True
    except Exception as e:
        print(f"Failed to load plugin {module_name}: {e}")
    return False


def get_installed_plugins() -> Dict[str, KiteMLPlugin]:
    return _PLUGINS
