"""
streaming.py — StreamPredictor for real-time record streaming inference in KiteML.
"""

from typing import Any, Iterable

import pandas as pd


class StreamPredictor:
    """
    Executes single-record streaming inference over record iterators.
    """

    def stream_predict(
        self,
        predictor_func: Any,
        records_iterator: Iterable[dict[str, Any]],
    ) -> list[Any]:
        """
        Run streaming prediction record-by-record.

        Parameters
        ----------
        predictor_func : Any
            Callable function taking a DataFrame and returning predictions.
        records_iterator : Iterable[dict[str, Any]]
            Iterator yielding record dictionaries.

        Returns
        -------
        list[Any]
            List of predicted output values.
        """
        predictions = []
        for record in records_iterator:
            single_df = pd.DataFrame([record])
            pred = predictor_func(single_df)
            val = pred[0] if hasattr(pred, "__getitem__") else pred
            predictions.append(val)
        return predictions
