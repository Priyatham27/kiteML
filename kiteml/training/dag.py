"""
dag.py — TrainingDAG and TrainingNode plugin architecture for KiteML training engine.
"""

import contextlib
from typing import Any

from kiteml.training.context import TrainingContext


class TrainingNode:
    """
    Base class for all training pipeline nodes.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def prepare(self, context: TrainingContext) -> None:
        """Prepare node resources and parameters."""

    def execute(self, context: TrainingContext) -> None:
        """Execute main node computation."""

    def validate(self, context: TrainingContext) -> bool:
        """Validate execution results."""
        return True

    def finalize(self, context: TrainingContext) -> None:
        """Finalize node execution."""

    def rollback(self, context: TrainingContext) -> None:
        """Rollback changes upon failure."""


class TrainingDAG:
    """
    Execution graph coordinating TrainingNodes in topological order.
    """

    def __init__(self) -> None:
        self.nodes: list[TrainingNode] = []

    def add_node(self, node: TrainingNode) -> None:
        """Add node to training DAG."""
        self.nodes.append(node)

    def execute_all(self, context: TrainingContext) -> TrainingContext:
        """
        Execute all nodes sequentially with validation and rollback safety.

        Parameters
        ----------
        context : TrainingContext
            Shared training context.

        Returns
        -------
        TrainingContext
            Updated context.
        """
        executed_nodes: list[TrainingNode] = []
        for node in self.nodes:
            try:
                node.prepare(context)
                node.execute(context)
                if not node.validate(context):
                    raise RuntimeError(f"Validation failed for node '{node.name}'.")
                node.finalize(context)
                executed_nodes.append(node)
            except Exception as e:
                for past_node in reversed(executed_nodes):
                    with contextlib.suppress(Exception):
                        past_node.rollback(context)
                raise e

        return context
