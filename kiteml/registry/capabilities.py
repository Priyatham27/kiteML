"""
capabilities.py — CapabilityAnalyzer Flagship Capability Scoring System for KiteML registry.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from kiteml.registry.model_info import ModelInfo


@dataclass
class CapabilityScore:
    """
    Score representation for algorithm dataset compatibility.
    """

    model_name: str
    score: float
    reasons: list[str]


class CapabilityAnalyzer:
    """
    Evaluates dataset characteristics and scores algorithm compatibility (0-100).
    """

    def score_model(
        self,
        info: ModelInfo,
        dataframe: pd.DataFrame,
        target_name: str | None = None,
        task_type: str = "classification",
    ) -> CapabilityScore:
        """
        Score a model's fit for the provided dataset.

        Parameters
        ----------
        info : ModelInfo
            Model metadata.
        dataframe : pd.DataFrame
            Target dataset.
        target_name : str, optional
            Target feature column.
        task_type : str
            ML task type.

        Returns
        -------
        CapabilityScore
            Scored capability record.
        """
        if not info.supports_task(task_type):
            return CapabilityScore(model_name=info.name, score=0.0, reasons=["Task type unsupported."])

        score = 80.0
        reasons = []

        n_samples = len(dataframe)
        n_features = len(dataframe.columns) - (1 if target_name and target_name in dataframe.columns else 0)

        # Family heuristics
        if info.family == "tree" or info.family == "ensemble" or info.family == "boosting":
            score += 10.0
            reasons.append("Ensemble/Tree models excel across diverse tabular data.")

        # Sample size penalties/boosts
        if n_samples > 10000 and info.family == "knn":
            score -= 30.0
            reasons.append("KNN scales poorly with large sample sizes.")
        elif n_samples > 10000 and info.family == "svm":
            score -= 20.0
            reasons.append("SVM training complexity is high on large datasets.")

        # High feature dimension
        if n_features > 50 and info.family in ["linear", "ensemble", "boosting"]:
            score += 5.0
            reasons.append("Well-suited for high-dimensional feature spaces.")

        final_score = max(0.0, min(100.0, score))
        return CapabilityScore(model_name=info.name, score=final_score, reasons=reasons)
