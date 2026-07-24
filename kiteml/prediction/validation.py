"""
validation.py — SchemaAdapter Flagship Intelligent Schema Adaptation Feature for Story 5.6.
"""

import warnings

import numpy as np
import pandas as pd


class SchemaAdapter:
    """
    Validates and adapts incoming inference DataFrames to match training schemas gracefully.
    """

    def adapt_schema(
        self,
        dataframe: pd.DataFrame,
        expected_columns: list[str],
        allow_missing: bool = True,
    ) -> pd.DataFrame:
        """
        Adapt inference DataFrame schema to match training features.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Incoming inference DataFrame.
        expected_columns : list[str]
            List of feature names expected by trained pipeline.
        allow_missing : bool
            Whether to fill missing columns with NaN.

        Returns
        -------
        pd.DataFrame
            Adapted DataFrame with matching columns and order.
        """
        if dataframe.empty:
            raise ValueError("Cannot run inference on an empty DataFrame.")

        df = dataframe.copy()
        current_cols = set(df.columns)
        expected_cols_set = set(expected_columns)

        # 1. Extra unexpected columns
        extra_cols = current_cols - expected_cols_set
        if extra_cols:
            warnings.warn(
                f"Inference DataFrame contains extra columns {list(extra_cols)} which will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            df = df.drop(columns=list(extra_cols))

        # 2. Missing columns
        missing_cols = expected_cols_set - set(df.columns)
        if missing_cols:
            if not allow_missing:
                raise ValueError(f"Missing required columns for inference: {list(missing_cols)}")
            warnings.warn(
                f"Inference DataFrame missing columns {list(missing_cols)}. Filling with NaN defaults.",
                UserWarning,
                stacklevel=2,
            )
            for col in missing_cols:
                df[col] = np.nan

        # 3. Reorder columns to match expected order
        return df[expected_columns]
