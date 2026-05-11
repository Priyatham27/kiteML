"""
cleaner.py - Handle missing values and basic data cleaning.
Updated for Pandas 3.x Copy-on-Write compliance.
"""

import pandas as pd


def handle_missing_values(df: pd.DataFrame, strategy: str = "auto") -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    strategy : str
        Strategy for handling missing values.
        - 'auto': median for numeric, mode for categorical.
        - 'drop': drop rows with missing values.
        - 'mean': fill numeric with mean, mode for categorical.
        - 'median': fill numeric with median, mode for categorical.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df = df.copy()

    if strategy == "drop":
        return df.dropna().reset_index(drop=True)

    numeric_cols = df.select_dtypes(include=["number"]).columns
    # Pandas 3.x: use ["object", "category", "str"] or just check non-numeric
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    # Fill numeric columns (CoW-safe assignment)
    for col in numeric_cols:
        if df[col].isnull().any():
            if strategy in ("auto", "median"):
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())

    # Fill categorical columns (CoW-safe assignment)
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_vals = df[col].mode()
            if len(mode_vals) > 0:
                df[col] = df[col].fillna(mode_vals[0])

    return df
