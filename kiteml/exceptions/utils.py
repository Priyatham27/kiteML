"""
utils.py — Helper functions for the KiteML exception framework.
"""

from typing import Any

import pandas as pd

from kiteml.exceptions.base import KiteMLError
from kiteml.exceptions.context import ErrorContext


def build_dataframe_context(
    df: Any,
    target: str | None = None,
    operation: str | None = None,
    feature_name: str | None = None,
    dataset_name: str | None = None,
    **kwargs: Any,
) -> ErrorContext:
    """
    Build an ErrorContext object from a DataFrame and workflow parameters.

    Parameters
    ----------
    df : Any
        DataFrame or data object.
    target : str, optional
        Target column name.
    operation : str, optional
        Current operation name.
    feature_name : str, optional
        Relevant feature column.
    dataset_name : str, optional
        Dataset path or name.
    **kwargs : Any
        Additional metadata.

    Returns
    -------
    ErrorContext
    """
    row_count = None
    column_count = None
    available_columns = None

    if isinstance(df, pd.DataFrame):
        row_count = len(df)
        column_count = len(df.columns)
        available_columns = [str(c) for c in df.columns]

    return ErrorContext(
        operation=operation,
        dataset_name=dataset_name,
        target=target,
        available_columns=available_columns,
        row_count=row_count,
        column_count=column_count,
        feature_name=feature_name,
        metadata=kwargs,
    )


def wrap_exception(
    exc: Exception,
    error_class: type[KiteMLError] = KiteMLError,
    message: str | None = None,
    **kwargs: Any,
) -> KiteMLError:
    """
    Wrap an existing Exception into a KiteMLError (or subclass), preserving cause.

    Parameters
    ----------
    exc : Exception
        Source exception to wrap.
    error_class : type of KiteMLError
        Exception class to instantiate.
    message : str, optional
        Custom error message. Defaults to str(exc).
    **kwargs : Any
        Keyword arguments passed to error_class constructor.

    Returns
    -------
    KiteMLError
    """
    msg = message if message is not None else str(exc)
    kml_exc = error_class(message=msg, **kwargs)
    kml_exc.__cause__ = exc
    return kml_exc
