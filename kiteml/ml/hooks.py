"""
hooks.py — MLHookRegistry for lifecycle hook execution in KiteML.
"""

import contextlib
from typing import Any, Callable


class MLHookRegistry:
    """
    Hook registry for lifecycle callbacks before and after ML workflow DAG stages.
    """

    def __init__(self) -> None:
        self.pre_hooks: dict[str, list[Callable[[Any], None]]] = {}
        self.post_hooks: dict[str, list[Callable[[Any], None]]] = {}

    def register_pre_hook(self, stage_name: str, callback: Callable[[Any], None]) -> None:
        """Register hook executed before stage."""
        self.pre_hooks.setdefault(stage_name, []).append(callback)

    def register_post_hook(self, stage_name: str, callback: Callable[[Any], None]) -> None:
        """Register hook executed after stage."""
        self.post_hooks.setdefault(stage_name, []).append(callback)

    def run_pre_hooks(self, stage_name: str, context: Any) -> None:
        """Execute pre-hooks."""
        for cb in self.pre_hooks.get(stage_name, []):
            with contextlib.suppress(Exception):
                cb(context)

    def run_post_hooks(self, stage_name: str, context: Any) -> None:
        """Execute post-hooks."""
        for cb in self.post_hooks.get(stage_name, []):
            with contextlib.suppress(Exception):
                cb(context)
