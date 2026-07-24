"""
hooks.py — HookRegistry lifecycle hook system for KiteML orchestration.
"""

import contextlib
from typing import Any, Callable


class HookRegistry:
    """
    Registry managing customizable lifecycle hook callbacks during pipeline execution.
    """

    SUPPORTED_HOOKS = {
        "before_validation",
        "after_validation",
        "before_preprocessing",
        "after_preprocessing",
        "before_transformation",
        "after_transformation",
        "before_serialization",
        "after_serialization",
    }

    def __init__(self) -> None:
        self._hooks: dict[str, list[Callable[..., Any]]] = {hook: [] for hook in self.SUPPORTED_HOOKS}

    def register_hook(self, hook_name: str, callback: Callable[..., Any]) -> None:
        """Register a callback function for a lifecycle hook."""
        if hook_name not in self.SUPPORTED_HOOKS:
            raise ValueError(f"Unsupported hook name '{hook_name}'. Supported hooks: {self.SUPPORTED_HOOKS}")
        self._hooks[hook_name].append(callback)

    def trigger(self, hook_name: str, context: Any = None) -> None:
        """Execute registered callbacks for the specified hook."""
        if hook_name in self._hooks:
            for cb in self._hooks[hook_name]:
                with contextlib.suppress(Exception):
                    cb(context)
