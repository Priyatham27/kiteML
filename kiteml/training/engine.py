"""
engine.py — TrainingEngine master entry point for KiteML model training subsystem.
"""

import time
from typing import Any

import pandas as pd

from kiteml.orchestration.orchestrator import KiteMLPipeline
from kiteml.training.context import TrainingContext
from kiteml.training.cross_validation import CrossValidationEngine
from kiteml.training.lifecycle import TrainingLifecycle
from kiteml.training.metrics import TrainingMetrics
from kiteml.training.session import TrainingResult, TrainingSession
from kiteml.training.splitter import DataSplitter
from kiteml.training.state import TrainingState
from kiteml.training.task_detector import TaskDetector
from kiteml.training.tracker import ExperimentTracker
from kiteml.training.trainer import ModelTrainer


class TrainingEngine:
    """
    Master Intelligent Model Training Engine orchestrating dataset splitting,
    task detection, pipeline transformation, cross-validation, and reproducibility tracking.
    """

    def __init__(self) -> None:
        self.task_detector = TaskDetector()
        self.splitter = DataSplitter()
        self.trainer = ModelTrainer()
        self.tracker = ExperimentTracker()

    def train(
        self,
        dataframe: pd.DataFrame,
        target: str,
        problem_type: str | None = None,
        random_state: int = 42,
        n_splits: int = 5,
        test_size: float = 0.2,
    ) -> TrainingResult:
        """
        Execute intelligent model training workflow on DataFrame.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Raw training DataFrame.
        target : str
            Target feature column name.
        problem_type : str, optional
            Explicit problem type override ('regression' or 'classification').
        random_state : int
            Random seed.
        n_splits : int
            Number of CV folds.
        test_size : float
            Test set proportion.

        Returns
        -------
        TrainingResult
            Result container with fitted model, metrics, and session metadata.
        """
        start_time = time.time()
        session = TrainingSession()
        lifecycle = TrainingLifecycle()
        lifecycle.transition_to(TrainingState.PREPARING)

        if target not in dataframe.columns:
            lifecycle.transition_to(TrainingState.FAILED, details=f"Target '{target}' not found.")
            raise ValueError(f"Target column '{target}' not found in DataFrame.")

        # 1. Run Epic 4 Pipeline Orchestration
        kiteml_pipeline = KiteMLPipeline()
        build_result = kiteml_pipeline.build(dataframe=dataframe, target=target, problem_type=problem_type)

        transformed_df = build_result.transformed_df
        target_series = dataframe[target]

        # 2. Task Detection
        inferred_task = problem_type or self.task_detector.detect_task(target_series)
        session.task_type = inferred_task

        # 3. Data Splitting
        lifecycle.transition_to(TrainingState.SPLITTING)
        X_train, X_test, y_train, y_test = self.splitter.split(
            X=transformed_df,
            y=target_series,
            test_size=test_size,
            random_state=random_state,
            task_type=inferred_task,
        )

        ctx = TrainingContext(
            dataset=dataframe,
            target_name=target,
            task_type=inferred_task,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            pipeline=build_result.pipeline,
        )

        # 4. Model Registry Discovery & Training
        lifecycle.transition_to(TrainingState.CROSS_VALIDATION)
        lifecycle.transition_to(TrainingState.TRAINING)

        from kiteml.registry import model_registry

        ranked_providers = model_registry.rank_models_for_dataset(
            dataframe=dataframe,
            target_name=target,
            task_type=inferred_task,
        )

        top_provider = ranked_providers[0][0] if ranked_providers else None
        candidate_model = top_provider.create() if top_provider else None

        fitted_model, cv_scores = self.trainer.train_model(
            X_train=X_train,
            y_train=y_train,
            task_type=inferred_task,
            n_splits=n_splits,
            random_state=random_state,
            model=candidate_model,
        )

        ctx.models.append(fitted_model)
        ctx.cv_scores[fitted_model.__class__.__name__] = cv_scores

        # 5. Metrics & Experiment Tracking
        lifecycle.transition_to(TrainingState.EVALUATION)
        exec_time = time.time() - start_time
        metrics = TrainingMetrics(
            training_time=exec_time,
            cpu_time=exec_time,
            n_samples=len(dataframe),
            n_features=len(X_train.columns),
            n_folds=n_splits,
        )

        exp_meta = self.tracker.track_experiment(ctx, random_state=random_state)

        session.finished_at = time.time()
        session.status = "COMPLETED"
        lifecycle.transition_to(TrainingState.COMPLETED)

        return TrainingResult(
            session=session,
            context=ctx,
            metrics=metrics,
            experiment_metadata=exp_meta,
            fitted_model=fitted_model,
        )
