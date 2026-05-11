"""
registry.py — Central Model Registry for KiteML.

This is the single source of truth for every model KiteML knows about.

Design
------
* Models are stored as plain dicts: name → unfitted estimator instance.
* Adding a new model requires ONE LINE here — nothing else changes.
* All random_state values come from kiteml.config.DEFAULT_RANDOM_STATE so
  changing the global seed is a single-file edit.
* get_classification_models() / get_regression_models() return *fresh*
  clones so repeated calls to select_best_model don't share state between
  cross-validation folds.

Usage (extending the registry)
-------------------------------
    from kiteml.models.registry import CLASSIFICATION_MODELS
    from xgboost import XGBClassifier
    CLASSIFICATION_MODELS["XGBoost"] = XGBClassifier(use_label_encoder=False)

That single line is all you need — the selector picks it up automatically.
"""

from sklearn.base import clone
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from kiteml.config import DEFAULT_RANDOM_STATE

RS = DEFAULT_RANDOM_STATE  # local shorthand keeps lines concise

# ---------------------------------------------------------------------------
# Classification Registry
# ---------------------------------------------------------------------------
#
# Key   → display name used in leaderboards and reports
# Value → unfitted scikit-learn compatible estimator
#
CLASSIFICATION_MODELS: dict = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RS),
    "RandomForest":       RandomForestClassifier(n_estimators=100, random_state=RS),
    "GradientBoosting":   GradientBoostingClassifier(n_estimators=100, random_state=RS),
    "DecisionTree":       DecisionTreeClassifier(random_state=RS),
    "SVM":                SVC(random_state=RS, probability=True),
    "KNN":                KNeighborsClassifier(),
    "AdaBoost":           AdaBoostClassifier(n_estimators=100, random_state=RS),
}

# ---------------------------------------------------------------------------
# Regression Registry
# ---------------------------------------------------------------------------
#
# Primary scoring metric : R²  (higher = better)
# Secondary report metric: RMSE (lower = better)
#
REGRESSION_MODELS: dict = {
    "LinearRegression": LinearRegression(),
    "Ridge":            Ridge(random_state=RS),
    "Lasso":            Lasso(random_state=RS),
    "ElasticNet":       ElasticNet(random_state=RS),
    "RandomForest":     RandomForestRegressor(n_estimators=100, random_state=RS),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=RS),
    "DecisionTree":     DecisionTreeRegressor(random_state=RS),
    "SVR":              SVR(),
    "KNN":              KNeighborsRegressor(),
    "AdaBoost":         AdaBoostRegressor(n_estimators=100, random_state=RS),
}


def get_classification_models() -> dict:
    """
    Return a fresh clone of every classification model in the registry.

    Cloning ensures each returned estimator is unfitted and independent —
    safe to pass directly into cross_val_score without state bleed.

    Returns
    -------
    dict
        ``{model_name: cloned_unfitted_estimator}``
    """
    return {name: clone(model) for name, model in CLASSIFICATION_MODELS.items()}


def get_regression_models() -> dict:
    """
    Return a fresh clone of every regression model in the registry.

    Returns
    -------
    dict
        ``{model_name: cloned_unfitted_estimator}``
    """
    return {name: clone(model) for name, model in REGRESSION_MODELS.items()}
