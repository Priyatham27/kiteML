"""
workflow_graph.py — MLWorkflowGraph Flagship ML DAG Execution Feature for Story 5.8.
"""

import contextlib
import time
from typing import Any

import numpy as np
import pandas as pd

from kiteml.deployment.engine import DeploymentEngine
from kiteml.evaluation.engine import EvaluationEngine
from kiteml.intelligence.problem_inference import infer_problem_type_advanced
from kiteml.ml.context import MLContext
from kiteml.ml.events import MLWorkflowEventBus
from kiteml.ml.hooks import MLHookRegistry
from kiteml.ml.metrics import UnifiedMetricsEngine
from kiteml.optimization.engine import OptimizationEngine
from kiteml.pipeline.transformation_pipeline import TransformationPipeline
from kiteml.registry import model_registry
from kiteml.selection.engine import ModelSelectionEngine
from kiteml.training.trainer import ModelTrainer


class MLWorkflowGraph:
    """
    Master ML DAG executor orchestrating dataset analysis, validation, transformation pipelines,
    model training, hyperparameter optimization, model evaluation, policy selection, and deployment packaging.
    """

    def __init__(self) -> None:
        self.event_bus = MLWorkflowEventBus()
        self.hook_registry = MLHookRegistry()
        self.metrics_engine = UnifiedMetricsEngine()

        self.trainer = ModelTrainer()
        self.optimization_engine = OptimizationEngine()
        self.evaluation_engine = EvaluationEngine()
        self.selection_engine = ModelSelectionEngine()
        self.deployment_engine = DeploymentEngine()

    def execute(
        self,
        context: MLContext,
        optimize_hyperparameters: bool = False,
        policy: str = "balanced",
    ) -> MLContext:
        """
        Execute end-to-end ML workflow DAG across shared MLContext.

        Parameters
        ----------
        context : MLContext
            Global shared state context.
        optimize_hyperparameters : bool
            Whether to run hyperparameter tuning.
        policy : str
            Active model selection policy ('balanced', 'accuracy', 'fast_inference', 'low_memory').

        Returns
        -------
        MLContext
            Updated context populated with all stage outputs.
        """
        self.metrics_engine.start_workflow()

        # ── 1. Analyze & Detect Problem Type ────────────────────────────────────
        t0 = time.time()
        self.hook_registry.run_pre_hooks("analyze", context)
        self.event_bus.publish("StageStarted", {"stage": "analyze"})

        target_series = context.dataframe[context.target_column]
        if pd.api.types.is_float_dtype(target_series) and not np.array_equal(
            target_series.dropna().values, np.floor(target_series.dropna().values)
        ):
            context.problem_type = "regression"
        else:
            inferred = infer_problem_type_advanced(target_series)
            context.problem_type = inferred.problem_type

        self.metrics_engine.record_stage_time("analyze", time.time() - t0)
        self.event_bus.publish("StageCompleted", {"stage": "analyze"})
        self.hook_registry.run_post_hooks("analyze", context)

        # ── 2. Build Pipeline & Transform Data ─────────────────────────────────
        t0 = time.time()
        self.hook_registry.run_pre_hooks("pipeline", context)
        self.event_bus.publish("StageStarted", {"stage": "pipeline"})

        X_raw = context.dataframe.drop(columns=[context.target_column])
        y_raw = context.dataframe[context.target_column]

        pipeline = TransformationPipeline()
        pipeline.fit(X_raw, y_raw)
        X_trans = pipeline.transform(X_raw)

        context.pipeline_result = pipeline
        self.metrics_engine.record_stage_time("pipeline", time.time() - t0)
        self.event_bus.publish("StageCompleted", {"stage": "pipeline"})
        self.hook_registry.run_post_hooks("pipeline", context)

        # ── 3. Train Candidate Models from Registry ───────────────────────────
        t0 = time.time()
        self.hook_registry.run_pre_hooks("train", context)
        self.event_bus.publish("StageStarted", {"stage": "train"})

        full_trans_df = pd.DataFrame(X_trans)
        full_trans_df["__target__"] = y_raw.values

        prob_str = str(context.problem_type).lower()
        reg_task = "regression" if "regress" in prob_str or "continuous" in prob_str else "classification"

        model_names = model_registry.list_models(task=reg_task)
        if not model_names:
            model_names = model_registry.list_models()

        candidate_evaluations: list[dict[str, Any]] = []

        n_samples = len(y_raw)
        n_splits = min(3, max(2, n_samples // 2)) if n_samples < 10 else 3

        for name in model_names:
            model_inst = model_registry.create(name)
            fitted_model, _ = self.trainer.train_model(
                X_train=X_trans,
                y_train=y_raw,
                task_type=reg_task,
                model=model_inst,
                n_splits=n_splits,
            )

            # ── 4. Hyperparameter Optimization (Optional) ──────────────────────
            opt_res = None
            if optimize_hyperparameters:
                with contextlib.suppress(Exception):
                    opt_res = self.optimization_engine.optimize(
                        model_name=name,
                        dataframe=full_trans_df,
                        target="__target__",
                        problem_type=context.problem_type,
                        max_trials=5,
                    )

            # ── 5. Evaluate Candidate Model ────────────────────────────────────
            eval_res = self.evaluation_engine.evaluate(
                model=fitted_model,
                X_test=X_trans,
                y_test=y_raw,
                problem_type=context.problem_type,
            )

            eval_metrics = eval_res.metrics if isinstance(eval_res.metrics, dict) else eval_res.metrics.to_dict()

            candidate_evaluations.append(
                {
                    "name": name,
                    "model": fitted_model,
                    "pipeline": pipeline,
                    "composite_score": eval_res.composite_score,
                    "metrics": eval_metrics,
                    "benchmark": eval_res.benchmark.to_dict(),
                    "evaluation": eval_res,
                    "optimization": opt_res,
                }
            )

        context.trained_models = candidate_evaluations
        self.metrics_engine.record_stage_time("train", time.time() - t0)
        self.event_bus.publish("StageCompleted", {"stage": "train"})
        self.hook_registry.run_post_hooks("train", context)

        # ── 6. Select Best Model ───────────────────────────────────────────────
        t0 = time.time()
        self.hook_registry.run_pre_hooks("select", context)
        self.event_bus.publish("StageStarted", {"stage": "select"})

        best_model = self.selection_engine.select(candidate_evaluations, policy=policy)
        context.best_model = best_model

        self.metrics_engine.record_stage_time("select", time.time() - t0)
        self.event_bus.publish("StageCompleted", {"stage": "select"})
        self.hook_registry.run_post_hooks("select", context)

        context.metrics = self.metrics_engine.get_summary()
        return context
