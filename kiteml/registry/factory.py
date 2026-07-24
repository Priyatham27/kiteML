"""
factory.py — ModelFactory for instantiating algorithm estimators in KiteML.
"""

from typing import Any

from kiteml.registry.providers import ModelProvider


class ModelFactory:
    """
    Factory instantiating model estimators from registered providers with configuration overrides.
    """

    def create_model(self, provider: ModelProvider, params: dict[str, Any] | None = None) -> Any:
        """
        Instantiate model from provider.

        Parameters
        ----------
        provider : ModelProvider
            Algorithm provider instance.
        params : dict[str, Any], optional
            Hyperparameter overrides.

        Returns
        -------
        Any
            Fitted/unfitted estimator object.
        """
        return provider.create(params=params)
