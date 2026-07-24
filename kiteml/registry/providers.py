"""
providers.py — ModelProvider interface and built-in algorithm providers for KiteML registry.
"""

import contextlib
from abc import ABC, abstractmethod
from typing import Any

from kiteml.registry.model_info import ModelInfo


class ModelProvider(ABC):
    """
    Abstract base class for all algorithm providers registered in KiteML.
    """

    @property
    @abstractmethod
    def info(self) -> ModelInfo:
        """Return model metadata."""
        pass

    @abstractmethod
    def create(self, params: dict[str, Any] | None = None) -> Any:
        """Instantiate algorithm estimator with optional parameter overrides."""
        pass


# =====================================================================
# Regression Providers
# =====================================================================


class LinearRegressionProvider(ModelProvider):

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="LinearRegression",
            family="linear",
            task_types=["regression"],
            tags=["linear", "fast", "interpretable"],
        )

    def create(self, params: dict[str, Any] | None = None) -> Any:
        from sklearn.linear_model import LinearRegression

        return LinearRegression(**(params or {}))


class RidgeProvider(ModelProvider):

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Ridge",
            family="linear",
            task_types=["regression"],
            tags=["linear", "regularized"],
        )

    def create(self, params: dict[str, Any] | None = None) -> Any:
        from sklearn.linear_model import Ridge

        return Ridge(**(params or {}))


class LassoProvider(ModelProvider):

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Lasso",
            family="linear",
            task_types=["regression"],
            tags=["linear", "sparse"],
        )

    def create(self, params: dict[str, Any] | None = None) -> Any:
        from sklearn.linear_model import Lasso

        return Lasso(**(params or {}))


class ElasticNetProvider(ModelProvider):

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="ElasticNet",
            family="linear",
            task_types=["regression"],
            tags=["linear", "regularized"],
        )

    def create(self, params: dict[str, Any] | None = None) -> Any:
        from sklearn.linear_model import ElasticNet

        return ElasticNet(**(params or {}))


class DecisionTreeRegressorProvider(ModelProvider):

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="DecisionTreeRegressor",
            family="tree",
            task_types=["regression"],
            tags=["tree", "interpretable"],
        )

    def create(self, params: dict[str, Any] | None = None) -> Any:
        from sklearn.tree import DecisionTreeRegressor

        return DecisionTreeRegressor(**(params or {}))


class RandomForestRegressorProvider(ModelProvider):

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="RandomForestRegressor",
            family="ensemble",
            task_types=["regression"],
            tags=["tree", "ensemble"],
        )

    def create(self, params: dict[str, Any] | None = None) -> Any:
        from sklearn.ensemble import RandomForestRegressor

        p = {"n_estimators": 50, "random_state": 42}
        if params:
            p.update(params)
        return RandomForestRegressor(**p)


class GradientBoostingRegressorProvider(ModelProvider):

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="GradientBoostingRegressor",
            family="boosting",
            task_types=["regression"],
            tags=["boosting", "ensemble"],
        )

    def create(self, params: dict[str, Any] | None = None) -> Any:
        from sklearn.ensemble import GradientBoostingRegressor

        p = {"n_estimators": 50, "random_state": 42}
        if params:
            p.update(params)
        return GradientBoostingRegressor(**p)


class ExtraTreesRegressorProvider(ModelProvider):

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="ExtraTreesRegressor",
            family="ensemble",
            task_types=["regression"],
            tags=["tree", "ensemble"],
        )

    def create(self, params: dict[str, Any] | None = None) -> Any:
        from sklearn.ensemble import ExtraTreesRegressor

        p = {"n_estimators": 50, "random_state": 42}
        if params:
            p.update(params)
        return ExtraTreesRegressor(**p)


# =====================================================================
# Classification Providers
# =====================================================================


class LogisticRegressionProvider(ModelProvider):

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="LogisticRegression",
            family="linear",
            task_types=["binary_classification", "multiclass_classification"],
            supports_probability=True,
            tags=["linear", "fast"],
        )

    def create(self, params: dict[str, Any] | None = None) -> Any:
        from sklearn.linear_model import LogisticRegression

        p = {"max_iter": 1000}
        if params:
            p.update(params)
        return LogisticRegression(**p)


class DecisionTreeClassifierProvider(ModelProvider):

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="DecisionTreeClassifier",
            family="tree",
            task_types=["binary_classification", "multiclass_classification"],
            supports_probability=True,
            tags=["tree"],
        )

    def create(self, params: dict[str, Any] | None = None) -> Any:
        from sklearn.tree import DecisionTreeClassifier

        return DecisionTreeClassifier(**(params or {}))


class RandomForestClassifierProvider(ModelProvider):

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="RandomForestClassifier",
            family="ensemble",
            task_types=["binary_classification", "multiclass_classification"],
            supports_probability=True,
            tags=["tree", "ensemble"],
        )

    def create(self, params: dict[str, Any] | None = None) -> Any:
        from sklearn.ensemble import RandomForestClassifier

        p = {"n_estimators": 50, "random_state": 42}
        if params:
            p.update(params)
        return RandomForestClassifier(**p)


class GradientBoostingClassifierProvider(ModelProvider):

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="GradientBoostingClassifier",
            family="boosting",
            task_types=["binary_classification", "multiclass_classification"],
            supports_probability=True,
            tags=["boosting", "ensemble"],
        )

    def create(self, params: dict[str, Any] | None = None) -> Any:
        from sklearn.ensemble import GradientBoostingClassifier

        p = {"n_estimators": 50, "random_state": 42}
        if params:
            p.update(params)
        return GradientBoostingClassifier(**p)


class ExtraTreesClassifierProvider(ModelProvider):

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="ExtraTreesClassifier",
            family="ensemble",
            task_types=["binary_classification", "multiclass_classification"],
            supports_probability=True,
            tags=["tree", "ensemble"],
        )

    def create(self, params: dict[str, Any] | None = None) -> Any:
        from sklearn.ensemble import ExtraTreesClassifier

        p = {"n_estimators": 50, "random_state": 42}
        if params:
            p.update(params)
        return ExtraTreesClassifier(**p)


class SVCProvider(ModelProvider):

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="SVC",
            family="svm",
            task_types=["binary_classification", "multiclass_classification"],
            supports_probability=True,
            tags=["svm"],
        )

    def create(self, params: dict[str, Any] | None = None) -> Any:
        from sklearn.svm import SVC

        p = {"probability": True, "random_state": 42}
        if params:
            p.update(params)
        return SVC(**p)


class KNeighborsClassifierProvider(ModelProvider):

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="KNeighborsClassifier",
            family="knn",
            task_types=["binary_classification", "multiclass_classification"],
            supports_probability=True,
            tags=["knn"],
        )

    def create(self, params: dict[str, Any] | None = None) -> Any:
        from sklearn.neighbors import KNeighborsClassifier

        return KNeighborsClassifier(**(params or {}))


def get_default_providers() -> list[ModelProvider]:
    """Get list of default built-in model providers."""
    providers: list[ModelProvider] = [
        LinearRegressionProvider(),
        RidgeProvider(),
        LassoProvider(),
        ElasticNetProvider(),
        DecisionTreeRegressorProvider(),
        RandomForestRegressorProvider(),
        GradientBoostingRegressorProvider(),
        ExtraTreesRegressorProvider(),
        LogisticRegressionProvider(),
        DecisionTreeClassifierProvider(),
        RandomForestClassifierProvider(),
        GradientBoostingClassifierProvider(),
        ExtraTreesClassifierProvider(),
        SVCProvider(),
        KNeighborsClassifierProvider(),
    ]
    return providers
