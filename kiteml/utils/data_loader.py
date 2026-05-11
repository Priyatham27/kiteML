"""
data_loader.py - Load datasets from various sources.
"""

from typing import Union

import pandas as pd


def load_data(data: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Load data from a file path or return the DataFrame directly.

    Parameters
    ----------
    data : str or pd.DataFrame
        Path to a CSV/Excel file, or a pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    ValueError
        If the data type is unsupported or file format is unrecognized.
    """
    if isinstance(data, pd.DataFrame):
        return data.copy()

    if isinstance(data, str):
        if data.endswith(".csv"):
            return pd.read_csv(data)
        elif data.endswith((".xls", ".xlsx")):
            return pd.read_excel(data)
        elif data.endswith(".json"):
            return pd.read_json(data)
        elif data.endswith(".parquet"):
            return pd.read_parquet(data)
        else:
            raise ValueError(
                f"Unsupported file format: {data}. " "Supported formats: .csv, .xls, .xlsx, .json, .parquet"
            )

    raise ValueError(f"Unsupported data type: {type(data)}. " "Pass a file path (str) or a pandas DataFrame.")
