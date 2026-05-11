"""
scaler.py - Feature scaling utilities.
"""

from typing import Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_features(X: pd.DataFrame, scaler: Optional[StandardScaler] = None) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Scale features using StandardScaler.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame.
    scaler : StandardScaler, optional
        A pre-fitted scaler. If None, a new one is fitted.

    Returns
    -------
    Tuple[pd.DataFrame, StandardScaler]
        Scaled DataFrame and the fitted scaler.
    """
    if scaler is None:
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(X)
    else:
        scaled_array = scaler.transform(X)

    scaled_df = pd.DataFrame(scaled_array, columns=X.columns, index=X.index)
    return scaled_df, scaler
