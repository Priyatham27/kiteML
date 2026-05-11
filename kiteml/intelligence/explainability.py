"""
explainability.py — Model explainability hooks for KiteML.

Extracts global feature importance from fitted models using:
  - feature_importances_ (tree ensembles, gradient boosting)
  - coef_ (linear models)
  - permutation importance (fallback for any estimator)

Also provides SHAP-ready hooks for future integration.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class FeatureImportanceEntry:
    feature: str
    importance: float
    rank: int
    direction: str  # "positive" | "negative" | "unknown"
    interpretation: str


@dataclass
class ExplainabilityReport:
    """Model explainability results."""

    model_name: str
    method: str  # "feature_importances_" | "coef_" | "permutation" | "unavailable"
    feature_importances: List[FeatureImportanceEntry]
    top_features: List[str]
    summary: str
    shap_ready: bool  # True if SHAP can be applied to this model type

    def to_dict(self) -> Dict[str, float]:
        return {e.feature: e.importance for e in self.feature_importances}


_SHAP_COMPATIBLE = {
    "RandomForestClassifier",
    "RandomForestRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "XGBClassifier",
    "XGBRegressor",
    "LGBMClassifier",
    "LGBMRegressor",
}


def explain_model(
    model: Any,
    feature_names: List[str],
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[Any] = None,
    top_n: int = 20,
) -> ExplainabilityReport:
    """
    Extract feature importance from a fitted model.

    Parameters
    ----------
    model : fitted estimator
    feature_names : list of str
    X_test : np.ndarray, optional
        Required for permutation importance fallback.
    y_test : array-like, optional
        Required for permutation importance fallback.
    top_n : int
        Number of top features to highlight.

    Returns
    -------
    ExplainabilityReport
    """
    model_name = type(model).__name__
    shap_ready = model_name in _SHAP_COMPATIBLE
    importances: List[FeatureImportanceEntry] = []
    method = "unavailable"

    raw: Optional[np.ndarray] = None

    # ── Method 1: feature_importances_ ───────────────────────────────────
    if hasattr(model, "feature_importances_"):
        raw = np.array(model.feature_importances_)
        method = "feature_importances_"

    # ── Method 2: coef_ ──────────────────────────────────────────────────
    elif hasattr(model, "coef_"):
        coef = np.array(model.coef_)
        if coef.ndim > 1:
            coef = coef[0]
        raw = coef
        method = "coef_"

    # ── Method 3: permutation importance (fallback) ───────────────────────
    elif X_test is not None and y_test is not None:
        try:
            from sklearn.inspection import permutation_importance

            result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
            raw = result.importances_mean
            method = "permutation"
        except Exception:
            pass

    if raw is not None and len(raw) == len(feature_names):
        abs_raw = np.abs(raw)
        sorted_idx = np.argsort(abs_raw)[::-1]

        for rank, idx in enumerate(sorted_idx[:top_n], start=1):
            feat = feature_names[idx]
            imp = float(raw[idx])
            abs_imp = float(abs_raw[idx])

            if method == "coef_":
                direction = "positive" if imp > 0 else "negative"
                interp = f"{'Increases' if imp > 0 else 'Decreases'} prediction by {abs_imp:.4f} per unit"
            elif method == "feature_importances_":
                direction = "unknown"
                interp = f"Explains {imp:.2%} of model decisions"
            else:
                direction = "unknown"
                interp = f"Permutation importance: {imp:.4f}"

            importances.append(
                FeatureImportanceEntry(
                    feature=feat,
                    importance=round(abs_imp, 6),
                    rank=rank,
                    direction=direction,
                    interpretation=interp,
                )
            )

        top_features = [e.feature for e in importances[:5]]
        summary = (
            f"Top predictor: '{top_features[0]}' " f"(importance={importances[0].importance:.4f}) via {method}."
            if top_features
            else "No feature importance available."
        )
    else:
        top_features = []
        summary = f"Feature importance not available for {model_name}."

    return ExplainabilityReport(
        model_name=model_name,
        method=method,
        feature_importances=importances,
        top_features=top_features,
        summary=summary,
        shap_ready=shap_ready,
    )
