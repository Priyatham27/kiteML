"""
preprocessing - Data cleaning, encoding, and scaling utilities.

The primary interface is the :class:`Preprocessor` pipeline which handles
imputation, one-hot encoding, and scaling in a single, leakage-free object.
Individual helpers (cleaner, encoder, scaler) remain available for
custom workflows.
"""

from kiteml.preprocessing.cleaner import handle_missing_values
from kiteml.preprocessing.encoder import encode_categoricals
from kiteml.preprocessing.pipeline import Preprocessor
from kiteml.preprocessing.scaler import scale_features
