"""
batch.py — BatchPredictor for chunked high-throughput batch inference in KiteML.
"""

from typing import Any

import numpy as np
import pandas as pd


class BatchPredictor:
    """
    Executes chunked batch inference over large DataFrames.
    """

    def batch_predict(
        self,
        predictor_func: Any,
        dataframe: pd.DataFrame,
        chunk_size: int = 1000,
    ) -> np.ndarray:
        """
        Run batch inference in chunks.

        Parameters
        ----------
        predictor_func : Any
            Callable function taking a DataFrame chunk and returning predictions.
        dataframe : pd.DataFrame
            Large input DataFrame.
        chunk_size : int
            Chunk size.

        Returns
        -------
        np.ndarray
            Aggregated predictions array.
        """
        n_rows = len(dataframe)
        if n_rows <= chunk_size:
            return np.asarray(predictor_func(dataframe))

        results = []
        for start_idx in range(0, n_rows, chunk_size):
            chunk = dataframe.iloc[start_idx : start_idx + chunk_size]
            preds = predictor_func(chunk)
            results.append(np.asarray(preds))

        return np.concatenate(results, axis=0)
