"""
workflow.py — WorkflowGraph stage runner orchestrating Epics 1–4 for KiteML.
"""

import time
from typing import Any

import pandas as pd

from kiteml.feature_engineering.planner import FeatureEngineeringEngine
from kiteml.feature_selection.planner import FeatureSelectionEngine
from kiteml.intelligence.data_profiler import build_data_profile
from kiteml.orchestration.context import OrchestrationContext
from kiteml.orchestration.events import OrchestrationEventBus
from kiteml.orchestration.hooks import HookRegistry
from kiteml.pipeline.dx_pipeline import DXPipeline
from kiteml.pipeline.transformation_pipeline import TransformationPipeline
from kiteml.preprocessing.planner import PreprocessingEngine


class WorkflowGraph:
    """
    Intelligent Workflow Graph orchestrating Intelligence, Validation, DX Diagnostics,
    Preprocessing, Feature Engineering, Selection, Transformation, and Reporting.
    """

    def __init__(
        self,
        hooks: HookRegistry | None = None,
        event_bus: OrchestrationEventBus | None = None,
    ) -> None:
        self.hooks = hooks or HookRegistry()
        self.event_bus = event_bus or OrchestrationEventBus()

    def run(self, context: OrchestrationContext) -> OrchestrationContext:
        """
        Execute the full AutoML transformation workflow.

        Parameters
        ----------
        context : OrchestrationContext
            Orchestration context holding dataset and parameters.

        Returns
        -------
        OrchestrationContext
            Updated orchestration context with blueprints, pipeline, and report.
        """
        df = context.dataset
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Invalid input DataFrame provided to WorkflowGraph.")

        target = context.target_name
        prob_type = context.problem_type or "classification"

        # 1. Validation & Intelligence Profiling
        self.hooks.trigger("before_validation", context)
        self.event_bus.emit("BeforeValidation", {"shape": df.shape})

        if target and target in df.columns:
            try:
                context.data_profile = build_data_profile(df, target=target, problem_type=prob_type)
            except Exception:
                context.data_profile = None

        dx_pipeline = DXPipeline()
        dx_pipeline.start()
        context.diagnostics = dx_pipeline.get_diagnostics()

        self.event_bus.emit("DatasetValidated", {"target": target})
        self.hooks.trigger("after_validation", context)

        # 2. Preprocessing & Feature Engineering Planning
        self.hooks.trigger("before_preprocessing", context)
        self.event_bus.emit("BeforePreprocessing")

        prep_engine = PreprocessingEngine()
        context.preprocessing_blueprint = prep_engine.plan(df, target=target, problem_type=prob_type)

        fe_engine = FeatureEngineeringEngine()
        context.engineering_blueprint = fe_engine.plan(
            df,
            preprocessing_blueprint=context.preprocessing_blueprint,
            target=target,
            problem_type=prob_type,
        )

        fs_engine = FeatureSelectionEngine()
        context.selection_blueprint = fs_engine.plan(
            df,
            preprocessing_blueprint=context.preprocessing_blueprint,
            feature_engineering_blueprint=context.engineering_blueprint,
            target=target,
            problem_type=prob_type,
            keep_features=context.keep_features,
        )

        self.event_bus.emit("BlueprintsGenerated")
        self.hooks.trigger("after_preprocessing", context)

        # 3. Transformation Pipeline Execution
        self.hooks.trigger("before_transformation", context)
        self.event_bus.emit("BeforeTransformation")

        pipeline = TransformationPipeline(
            preprocessing_blueprint=context.preprocessing_blueprint,
            engineering_blueprint=context.engineering_blueprint,
            selection_blueprint=context.selection_blueprint,
        )

        context.transformed_df = pipeline.fit_transform(
            df,
            target=target,
            problem_type=prob_type,
            keep_features=context.keep_features,
        )

        context.pipeline = pipeline
        context.report = pipeline.report()

        self.event_bus.emit("TransformationCompleted", {"final_shape": context.transformed_df.shape})
        self.hooks.trigger("after_transformation", context)

        return context
