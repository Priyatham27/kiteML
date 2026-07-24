"""
result.py — TrainingResult container for Story 5.8 flagship entry point in KiteML.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kiteml.deployment.engine import DeploymentEngine
from kiteml.ml.report import UnifiedReport
from kiteml.prediction.engine import PredictionEngine, PredictionResult
from kiteml.selection.best_model import BestModel


@dataclass
class TrainingResult:
    """
    High-level TrainingResult object returned by kiteml.train().
    """

    best_model: BestModel
    pipeline: Any
    problem_type: str
    report: UnifiedReport
    metrics: dict[str, Any]

    def predict(self, dataframe: Any) -> PredictionResult:
        """
        Execute prediction on new inference DataFrame.

        Parameters
        ----------
        dataframe : Any
            Inference DataFrame.

        Returns
        -------
        PredictionResult
            Inference result container.
        """
        engine = PredictionEngine()
        return engine.predict(
            model=self.best_model.model,
            dataframe=dataframe,
            pipeline=self.pipeline,
        )

    def predict_proba(self, dataframe: Any) -> Any:
        """
        Execute prediction probabilities on inference DataFrame.
        """
        res = self.predict(dataframe)
        if res.probabilities is not None:
            return res.probabilities
        raise AttributeError(f"Winning model '{self.best_model.model_name}' does not support predict_proba().")

    def save(self, path: Path | str) -> Any:
        """
        Package and save full training solution into .kiteml deployment archive.

        Parameters
        ----------
        path : Path | str
            Destination file path.

        Returns
        -------
        DeploymentReport
            Deployment packaging summary report.
        """
        engine = DeploymentEngine()
        return engine.package(
            model=self.best_model.model,
            model_name=self.best_model.model_name,
            task_type=self.problem_type,
            output_path=path,
            pipeline=self.pipeline,
        )

    def summary(self, width: int = 55) -> str:
        """Render terminal report summary."""
        return self.report.summary(width=width)
