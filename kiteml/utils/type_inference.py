"""
type_inference.py - Detect whether the problem is classification or regression.
"""

import pandas as pd


def infer_problem_type(df: pd.DataFrame, target: str, threshold: int = 20) -> str:
    """
    Infer whether the task is classification or regression based on the target column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target : str
        Name of the target column.
    threshold : int
        If the number of unique values in the target is <= threshold,
        treat it as classification. Default 20.

    Returns
    -------
    str
        'classification' or 'regression'.
    """
    target_series = df[target]

    # If target is object/category, it's classification
    if target_series.dtype == "object" or target_series.dtype.name == "category":
        return "classification"

    # If target is bool, it's classification
    if target_series.dtype == "bool":
        return "classification"

    # If few unique values, treat as classification
    n_unique = target_series.nunique()
    if n_unique <= threshold:
        return "classification"

    return "regression"
