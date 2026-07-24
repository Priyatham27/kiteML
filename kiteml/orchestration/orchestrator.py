"""
orchestrator.py — KiteMLPipeline and PipelineBuildResult master API for KiteML orchestration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from kiteml.orchestration.context import OrchestrationContext
from kiteml.orchestration.events import OrchestrationEventBus
from kiteml.orchestration.hooks import HookRegistry
from kiteml.orchestration.workflow import WorkflowGraph
from kiteml.pipeline.transformation_pipeline import TransformationPipeline


@dataclass
class PipelineBuildResult:
    """
    Result container holding pipeline artifacts, report, diagnostics, and transformed dataset.
    """

    pipeline: TransformationPipeline
    report: Any
    diagnostics: Any
    metrics: Any
    transformed_df: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


class KiteMLPipeline:
    """
    Master AutoML Pipeline Orchestrator.

    Unifies Intelligence profiling, Validation, DX Diagnostics, Preprocessing,
    Feature Engineering, Selection, Transformation, Reporting, and Serialization.
    """

    def __init__(self) -> None:
        self.hooks = HookRegistry()
        self.event_bus = OrchestrationEventBus()
        self.workflow = WorkflowGraph(hooks=self.hooks, event_bus=self.event_bus)
        self.pipeline: TransformationPipeline | None = None
        self.last_result: PipelineBuildResult | None = None

    def add_hook(self, hook_name: str, callback: Callable[..., Any]) -> None:
        """Register a custom lifecycle callback hook."""
        self.hooks.register_hook(hook_name, callback)

    def build(
        self,
        dataframe: pd.DataFrame,
        target: str | None = None,
        problem_type: str | None = None,
        keep_features: Sequence[str] | None = None,
    ) -> PipelineBuildResult:
        """
        Execute full AutoML pipeline build on input DataFrame.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Raw dataset.
        target : str, optional
            Target feature column name.
        problem_type : str, optional
            Task type ('classification' or 'regression').
        keep_features : Sequence[str], optional
            Protected features to preserve during selection.

        Returns
        -------
        PipelineBuildResult
            Result container with pipeline, report, diagnostics, metrics, and transformed_df.
        """
        ctx = OrchestrationContext(
            dataset=dataframe,
            target_name=target,
            problem_type=problem_type,
            keep_features=list(keep_features) if keep_features else [],
        )

        completed_ctx = self.workflow.run(ctx)
        if completed_ctx.pipeline is None:
            raise RuntimeError("Pipeline failed to initialize during workflow run.")

        self.pipeline = completed_ctx.pipeline

        result = PipelineBuildResult(
            pipeline=completed_ctx.pipeline,
            report=completed_ctx.report,
            diagnostics=completed_ctx.diagnostics,
            metrics=getattr(completed_ctx.report, "statistics", None),
            transformed_df=completed_ctx.transformed_df if completed_ctx.transformed_df is not None else pd.DataFrame(),
            metadata={
                "target_name": target,
                "problem_type": problem_type,
            },
        )
        self.last_result = result
        return result

    def save(self, filepath: str | Path) -> str:
        """Save pipeline to a .kml package file."""
        if not self.pipeline:
            raise RuntimeError("KiteMLPipeline is not built yet. Call build() before save().")
        self.hooks.trigger("before_serialization", self.pipeline)
        path = self.pipeline.save(filepath)
        self.hooks.trigger("after_serialization", path)
        return path

    @classmethod
    def load(cls, filepath: str | Path) -> "KiteMLPipeline":
        """Load fitted KiteMLPipeline from a .kml package file."""
        pipeline_instance = cls()
        pipeline_instance.pipeline = TransformationPipeline.load(filepath)
        return pipeline_instance
