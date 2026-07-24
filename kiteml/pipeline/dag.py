"""
dag.py — PipelineDAG Directed Acyclic Graph execution engine for KiteML.
"""

from typing import Any, Sequence


class DAGNode:
    """Node in the PipelineDAG representing a transformation stage and its dependencies."""

    def __init__(self, stage: Any, depends_on: list[str] | None = None) -> None:
        self.stage = stage
        self.name: str = getattr(stage, "name", type(stage).__name__)
        self.depends_on: list[str] = depends_on or []


class PipelineDAG:
    """
    Directed Acyclic Graph (DAG) for orchestrating transformation pipeline stages.

    Manages dependency resolution, topological sorting, conditional skipping,
    and stage execution order.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, DAGNode] = {}

    def add_stage(self, stage: Any, depends_on: Sequence[str] | None = None) -> None:
        """Add a stage node to the DAG."""
        node = DAGNode(stage=stage, depends_on=list(depends_on) if depends_on else [])
        self.nodes[node.name] = node

    def topological_sort(self) -> list[Any]:
        """
        Perform topological sort on registered DAG nodes.

        Returns
        -------
        list[PipelineStage]
            List of stages ordered by priority and dependency topology.
        """
        in_degree: dict[str, int] = {name: 0 for name in self.nodes}
        graph: dict[str, list[str]] = {name: [] for name in self.nodes}

        for name, node in self.nodes.items():
            for dep in node.depends_on:
                if dep in self.nodes:
                    graph[dep].append(name)
                    in_degree[name] += 1

        # Queue nodes with 0 in-degree, sorted by stage priority if present
        queue: list[str] = [name for name, deg in in_degree.items() if deg == 0]
        queue.sort(key=lambda n: getattr(self.nodes[n].stage, "priority", 50))

        ordered_stages: list[Any] = []
        visited_count = 0

        while queue:
            curr_name = queue.pop(0)
            ordered_stages.append(self.nodes[curr_name].stage)
            visited_count += 1

            for neighbor in graph[curr_name]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    queue.sort(key=lambda n: getattr(self.nodes[n].stage, "priority", 50))

        if visited_count != len(self.nodes):
            # Fallback to priority sort if cycle detected
            return sorted(
                [node.stage for node in self.nodes.values()],
                key=lambda s: getattr(s, "priority", 50),
            )

        return ordered_stages
