"""
realtime_inference.py — Low-latency single-record inference for KiteML.

Wraps a loaded bundle or Result with caching, schema validation, and
preprocessing reuse for fast production predictions.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from kiteml.deployment.inference_guardrails import InferenceGuardrails


@dataclass
class PredictionResult:
    """Result of a single inference call."""
    prediction: Any
    probabilities: Optional[Dict] = None   # class → probability (classification)
    latency_ms: float = 0.0
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    def __str__(self) -> str:
        out = f"Prediction: {self.prediction}"
        if self.probabilities:
            out += f"  |  Proba: {self.probabilities}"
        out += f"  |  Latency: {self.latency_ms:.2f}ms"
        return out


class RealtimeInferenceEngine:
    """
    Fast single-record inference engine with preprocessing caching.

    Parameters
    ----------
    model : fitted sklearn estimator
    preprocessor : fitted KiteML Preprocessor, optional
    feature_names : list of str
    problem_type : str
    schema : dict, optional
        Schema from .kiteml bundle for guardrail validation.
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        problem_type: str,
        preprocessor: Optional[Any] = None,
        schema: Optional[Dict] = None,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.problem_type = problem_type
        self._guardrails = InferenceGuardrails(feature_names, schema)
        self._call_count = 0
        self._total_latency_ms = 0.0

    @classmethod
    def from_bundle(cls, bundle_path: str) -> "RealtimeInferenceEngine":
        """Load inference engine from a .kiteml bundle directory."""
        from kiteml.deployment.packaging import load_bundle
        bundle = load_bundle(bundle_path)
        meta = bundle.get("metadata", {})
        return cls(
            model=bundle["model"],
            feature_names=meta.get("feature_names", []),
            problem_type=meta.get("problem_type", "classification"),
            preprocessor=bundle.get("preprocessor"),
            schema=bundle.get("schema"),
        )

    def predict(
        self,
        X: Union[Dict, pd.DataFrame, np.ndarray],
        validate: bool = True,
    ) -> PredictionResult:
        """
        Run inference on a single record or small batch.

        Parameters
        ----------
        X : dict, DataFrame, or ndarray
        validate : bool
            If True, run schema validation before inference.

        Returns
        -------
        PredictionResult
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        # ── Validate ──────────────────────────────────────────────────────
        if validate:
            guard_result = self._guardrails.validate(X)
            if not guard_result.is_valid:
                guard_result.raise_if_invalid()
            warnings = [v.message for v in guard_result.warnings]

        # ── Normalize input ───────────────────────────────────────────────
        if isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            df = pd.DataFrame(X, columns=self.feature_names[:X.shape[1]])
        else:
            df = pd.DataFrame(X)

        # Keep only expected columns
        present = [f for f in self.feature_names if f in df.columns]
        df = df[present]

        # ── Preprocess ────────────────────────────────────────────────────
        if self.preprocessor is not None:
            X_transformed = self.preprocessor.transform(df)
        else:
            X_transformed = df.values

        # ── Predict ───────────────────────────────────────────────────────
        raw_pred = self.model.predict(X_transformed)
        prediction = raw_pred[0] if len(raw_pred) == 1 else raw_pred.tolist()

        # ── Probabilities (classification) ────────────────────────────────
        probabilities = None
        if self.problem_type == "classification" and hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_transformed)
            classes = (
                [str(c) for c in self.model.classes_]
                if hasattr(self.model, "classes_") else
                [str(i) for i in range(proba.shape[1])]
            )
            probabilities = {cls: round(float(p), 4) for cls, p in zip(classes, proba[0])}

        latency_ms = (time.perf_counter() - t0) * 1000
        self._call_count += 1
        self._total_latency_ms += latency_ms

        return PredictionResult(
            prediction=prediction,
            probabilities=probabilities,
            latency_ms=round(latency_ms, 3),
            warnings=warnings,
        )

    @property
    def stats(self) -> Dict:
        """Runtime inference statistics."""
        avg = self._total_latency_ms / self._call_count if self._call_count > 0 else 0.0
        return {
            "total_calls": self._call_count,
            "total_latency_ms": round(self._total_latency_ms, 2),
            "avg_latency_ms": round(avg, 2),
        }
