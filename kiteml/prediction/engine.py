"""
engine.py — PredictionEngine and PredictionResult master entry point for KiteML prediction subsystem.
"""

import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from kiteml.prediction.batch import BatchPredictor
from kiteml.prediction.confidence import ConfidenceEngine
from kiteml.prediction.pipeline import PipelineReplayEngine
from kiteml.prediction.report import PredictionReport
from kiteml.prediction.streaming import StreamPredictor
from kiteml.prediction.validation import SchemaAdapter


@dataclass
class PredictionResult:
    """
    Result container returned by PredictionEngine.predict().
    """

    predictions: np.ndarray
    probabilities: Any | None = None
    confidence: np.ndarray | None = None
    report: PredictionReport | None = None

    def __getitem__(self, item: Any) -> Any:
        return self.predictions[item]

    def __len__(self) -> int:
        return len(self.predictions)


class PredictionEngine:
    """
    Master Intelligent Prediction & Inference Engine executing pipeline replay and model prediction.
    """

    def __init__(self) -> None:
        self.schema_adapter = SchemaAdapter()
        self.replay_engine = PipelineReplayEngine()
        self.confidence_engine = ConfidenceEngine()
        self.batch_predictor = BatchPredictor()
        self.stream_predictor = StreamPredictor()

    def predict(
        self,
        model: Any,
        dataframe: pd.DataFrame,
        pipeline: Any | None = None,
        expected_columns: list[str] | None = None,
    ) -> PredictionResult:
        """
        Execute end-to-end inference on incoming DataFrame.

        Parameters
        ----------
        model : Any
            Fitted estimator model.
        dataframe : pd.DataFrame
            Raw input DataFrame.
        pipeline : Any, optional
            Fitted transformation pipeline instance.
        expected_columns : list[str], optional
            List of expected feature column names for schema adaptation.

        Returns
        -------
        PredictionResult
            Inference prediction result object.
        """
        start_time = time.time()

        exp_cols = expected_columns
        if exp_cols is None and pipeline is not None and getattr(pipeline, "feature_names", None):
            exp_cols = list(pipeline.feature_names)
        elif exp_cols is None and pipeline is not None and getattr(pipeline, "feature_names_in_", None) is not None:
            exp_cols = list(pipeline.feature_names_in_)
        elif exp_cols is None:
            exp_cols = list(dataframe.columns)

        adapted_df = self.schema_adapter.adapt_schema(dataframe, expected_columns=exp_cols)
        X_trans = self.replay_engine.replay(pipeline, adapted_df)

        preds = model.predict(X_trans)
        proba, conf = self.confidence_engine.compute_confidence(model, X_trans)

        exec_dur = time.time() - start_time
        avg_conf = float(np.mean(conf)) if conf is not None else None

        report = PredictionReport(
            n_samples=len(dataframe),
            has_probabilities=proba is not None,
            average_confidence=avg_conf,
            execution_time_sec=exec_dur,
        )

        return PredictionResult(
            predictions=np.asarray(preds),
            probabilities=proba,
            confidence=conf,
            report=report,
        )

    def batch_predict(
        self,
        model: Any,
        dataframe: pd.DataFrame,
        pipeline: Any | None = None,
        chunk_size: int = 1000,
    ) -> PredictionResult:
        """
        Run high-throughput chunked batch inference.

        Parameters
        ----------
        model : Any
            Fitted estimator model.
        dataframe : pd.DataFrame
            Large input DataFrame.
        pipeline : Any, optional
            Fitted transformation pipeline instance.
        chunk_size : int
            Chunk size.

        Returns
        -------
        PredictionResult
            Inference prediction result.
        """

        def _chunk_func(df_chunk: pd.DataFrame) -> np.ndarray:
            res = self.predict(model=model, dataframe=df_chunk, pipeline=pipeline)
            return res.predictions

        batch_preds = self.batch_predictor.batch_predict(_chunk_func, dataframe, chunk_size=chunk_size)
        return PredictionResult(predictions=batch_preds)

    def stream_predict(
        self,
        model: Any,
        records_iterator: Iterable[dict[str, Any]],
        pipeline: Any | None = None,
    ) -> list[Any]:
        """
        Run continuous record-by-record streaming prediction.

        Parameters
        ----------
        model : Any
            Fitted estimator model.
        records_iterator : Iterable[dict[str, Any]]
            Record iterator.
        pipeline : Any, optional
            Fitted transformation pipeline instance.

        Returns
        -------
        list[Any]
            List of predicted output values.
        """

        def _single_func(df_single: pd.DataFrame) -> np.ndarray:
            res = self.predict(model=model, dataframe=df_single, pipeline=pipeline)
            return res.predictions

        return self.stream_predictor.stream_predict(_single_func, records_iterator)
