"""
pipeline.py — PipelineReplayEngine for executing saved transformation pipelines in KiteML.
"""

from typing import Any

import pandas as pd


class PipelineReplayEngine:
    """
    Replays trained Epic 4 transformation pipelines on inference datasets.
    """

    def replay(self, pipeline: Any, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Transform inference DataFrame using trained pipeline object.

        Parameters
        ----------
        pipeline : Any
            Fitted pipeline instance.
        dataframe : pd.DataFrame
            Adapted inference DataFrame.

        Returns
        -------
        pd.DataFrame
            Transformed feature matrix ready for model prediction.
        """
        if pipeline is None:
            return dataframe

        if hasattr(pipeline, "transform"):
            res = pipeline.transform(dataframe)
            if isinstance(res, pd.DataFrame):
                return res
            return pd.DataFrame(res)

        return dataframe
