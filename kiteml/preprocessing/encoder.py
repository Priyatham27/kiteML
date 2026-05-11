"""
encoder.py - Categorical feature encoding.
"""


import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_categoricals(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Encode categorical columns using LabelEncoder.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target : str
        Name of the target column (will also be encoded if categorical).

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, LabelEncoder]]
        Encoded DataFrame and a dict mapping column names to fitted encoders.
    """
    df = df.copy()
    encoders = {}

    categorical_cols = df.select_dtypes(include=["object", "category", "string", "str"]).columns

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders
