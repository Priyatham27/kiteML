"""
engine.py — OptimizationEngine master entry point for hyperparameter tuning in KiteML.
"""

import time
from typing import Any

import pandas as pd

from kiteml.optimization.advisor import OptimizationAdvisor
from kiteml.optimization.early_stopping import EarlyStopping
from kiteml.optimization.metrics import OptimizationMetrics
from kiteml.optimization.search_space import SearchSpace
from kiteml.optimization.session import OptimizationResult, OptimizationSession
from kiteml.optimization.trials import OptimizationTrial, TrialManager
from kiteml.registry import model_registry
from kiteml.training.trainer import ModelTrainer


class OptimizationEngine:
    """
    Master Hyperparameter Optimization Engine executing model-agnostic tuning search.
    """

    def __init__(self) -> None:
        self.advisor = OptimizationAdvisor()

    def optimize(
        self,
        model_name: str,
        dataframe: pd.DataFrame,
        target: str,
        problem_type: str | None = None,
        search_space: SearchSpace | None = None,
        max_trials: int = 10,
        time_limit: float = 300.0,
        random_state: int = 42,
    ) -> OptimizationResult:
        """
        Execute hyperparameter optimization search for specified model algorithm.

        Parameters
        ----------
        model_name : str
            Registered model name.
        dataframe : pd.DataFrame
            Dataset DataFrame.
        target : str
            Target feature column name.
        problem_type : str, optional
            Task type ('classification' or 'regression').
        search_space : SearchSpace, optional
            Custom parameter search space.
        max_trials : int
            Maximum trial evaluations limit.
        time_limit : float
            Time budget limit in seconds.
        random_state : int
            Random seed.

        Returns
        -------
        OptimizationResult
            Optimization result with best parameters and trial logs.
        """
        start_time = time.time()
        session = OptimizationSession(model_name=model_name)
        trial_mgr = TrialManager()
        stopping = EarlyStopping(max_trials=max_trials, time_limit=time_limit)

        s_space = search_space or SearchSpace.get_default_search_space(model_name)

        strategy, explanation = self.advisor.select_strategy(
            search_space=s_space,
            n_samples=len(dataframe),
            max_trials=max_trials,
        )
        session.strategy_name = strategy.__class__.__name__

        combos = strategy.generate_parameter_combinations(s_space, max_trials=max_trials, random_state=random_state)

        task_type = problem_type or ("classification" if dataframe[target].nunique() <= 20 else "regression")
        X = dataframe.drop(columns=[target])
        y = dataframe[target]

        trainer = ModelTrainer()

        for idx, params in enumerate(combos, 1):
            t0 = time.time()
            try:
                model_inst = model_registry.create(model_name, params=params)
                _, cv_scores = trainer.train_model(
                    X_train=X,
                    y_train=y,
                    task_type=task_type,
                    n_splits=3,
                    random_state=random_state,
                    model=model_inst,
                )
                score = float(sum(cv_scores) / len(cv_scores)) if cv_scores else 0.0
                status = "COMPLETED"
            except Exception:
                score = -999.0
                status = "FAILED"

            dur = time.time() - t0
            trial = OptimizationTrial(trial_id=idx, parameters=params, score=score, duration=dur, status=status)
            trial_mgr.record_trial(trial)

            elapsed = time.time() - start_time
            if stopping.should_stop(trial_count=idx, elapsed_time=elapsed, current_score=score):
                break

        best_trial = trial_mgr.get_best_trial()
        best_params = best_trial.parameters if best_trial else {}
        best_score = best_trial.score if best_trial else 0.0

        session.finished_at = time.time()
        session.status = "COMPLETED"

        opt_time = time.time() - start_time
        metrics = OptimizationMetrics(
            n_trials=len(trial_mgr.trials),
            best_score=best_score,
            optimization_time=opt_time,
        )

        return OptimizationResult(
            best_parameters=best_params,
            best_score=best_score,
            best_trial=best_trial,
            session=session,
            trials=trial_mgr.trials,
            metrics=metrics,
        )
