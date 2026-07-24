"""
preprocessing - Data cleaning, encoding, scaling, and intelligent preprocessing planning utilities.
"""

from kiteml.preprocessing.blueprint import FeaturePlan, PreprocessingBlueprint
from kiteml.preprocessing.cleaner import handle_missing_values
from kiteml.preprocessing.encoder import encode_categoricals
from kiteml.preprocessing.engine import PreprocessingEngine
from kiteml.preprocessing.pipeline import Preprocessor
from kiteml.preprocessing.providers import (
    BaseStrategyProvider,
    DatetimeProvider,
    EncodingProvider,
    MissingValueProvider,
    ScalingProvider,
    TextProvider,
)
from kiteml.preprocessing.rules import RuleEngine
from kiteml.preprocessing.scaler import scale_features
from kiteml.preprocessing.strategy import (
    DatetimeStrategy,
    EncodingStrategy,
    MissingStrategy,
    ScalingStrategy,
    TextStrategy,
)

__all__ = [
    "Preprocessor",
    "handle_missing_values",
    "encode_categoricals",
    "scale_features",
    "PreprocessingEngine",
    "PreprocessingBlueprint",
    "FeaturePlan",
    "RuleEngine",
    "MissingStrategy",
    "EncodingStrategy",
    "ScalingStrategy",
    "DatetimeStrategy",
    "TextStrategy",
    "BaseStrategyProvider",
    "MissingValueProvider",
    "EncodingProvider",
    "ScalingProvider",
    "DatetimeProvider",
    "TextProvider",
]
