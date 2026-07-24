"""
registry.py — ModelRegistry master catalog singleton for KiteML algorithm discovery.
"""

from typing import Any

import pandas as pd

from kiteml.registry.capabilities import CapabilityAnalyzer
from kiteml.registry.factory import ModelFactory
from kiteml.registry.providers import ModelProvider, get_default_providers


class ModelRegistry:
    """
    Central catalog storing, discovering, and instantiating ML algorithm providers.
    """

    def __init__(self) -> None:
        self._providers: dict[str, ModelProvider] = {}
        self.factory = ModelFactory()
        self.analyzer = CapabilityAnalyzer()
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Load default built-in providers into registry."""
        for provider in get_default_providers():
            self.register(provider)

    def register(self, provider: ModelProvider) -> None:
        """Register a model provider."""
        self._providers[provider.info.name] = provider

    def unregister(self, name: str) -> None:
        """Unregister a model provider by name."""
        if name in self._providers:
            del self._providers[name]

    def get(self, name: str) -> ModelProvider:
        """Retrieve model provider by name."""
        if name not in self._providers:
            raise KeyError(
                f"Model '{name}' not found in ModelRegistry. Registered models: {list(self._providers.keys())}"
            )
        return self._providers[name]

    def list_models(self, task: str | None = None, family: str | None = None) -> list[str]:
        """List registered model names filtered by task or family."""
        res = []
        for name, provider in self._providers.items():
            info = provider.info
            if task and not info.supports_task(task):
                continue
            if family and info.family != family:
                continue
            res.append(name)
        return res

    def search(self, tags: list[str]) -> list[str]:
        """Find model names containing specified tags."""
        res = []
        for name, provider in self._providers.items():
            if any(t in provider.info.tags for t in tags):
                res.append(name)
        return res

    def create(self, name: str, params: dict[str, Any] | None = None) -> Any:
        """Create a model instance by name."""
        provider = self.get(name)
        return self.factory.create_model(provider, params=params)

    def rank_models_for_dataset(
        self,
        dataframe: pd.DataFrame,
        target_name: str | None = None,
        task_type: str = "classification",
    ) -> list[tuple[ModelProvider, float]]:
        """
        Rank candidate model providers by capability score for the dataset.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Target dataset.
        target_name : str, optional
            Target feature column name.
        task_type : str
            ML task type.

        Returns
        -------
        list[tuple[ModelProvider, float]]
            List of (ModelProvider, score) tuples sorted by descending score.
        """
        scored = []
        for provider in self._providers.values():
            if provider.info.supports_task(task_type):
                cap_score = self.analyzer.score_model(
                    info=provider.info,
                    dataframe=dataframe,
                    target_name=target_name,
                    task_type=task_type,
                )
                scored.append((provider, cap_score.score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


# Global master model registry singleton
model_registry = ModelRegistry()
