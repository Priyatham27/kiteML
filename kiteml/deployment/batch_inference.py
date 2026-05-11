"""
batch_inference.py — Memory-safe batch prediction on large datasets.

Supports chunked streaming inference over CSVs, Parquet files, and DataFrames
with progress tracking and optional output writing.
"""

import time
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class BatchResult:
    """Result of a batch inference operation."""

    predictions: np.ndarray
    probabilities: Optional[np.ndarray]  # shape (n, n_classes) or None
    n_rows: int
    n_chunks: int
    elapsed_s: float
    throughput_rps: float  # rows per second

    def to_dataframe(self) -> pd.DataFrame:
        """Return predictions as a DataFrame (with probabilities if available)."""
        df = pd.DataFrame({"prediction": self.predictions})
        if self.probabilities is not None:
            for i in range(self.probabilities.shape[1]):
                df[f"proba_class_{i}"] = self.probabilities[:, i]
        return df

    def save_csv(self, path: str) -> str:
        """Save predictions to CSV."""
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"✅ Batch predictions saved → {path}")
        return path


def batch_predict(
    result: Any,
    data: Union[str, pd.DataFrame],
    chunk_size: int = 1000,
    output_path: Optional[str] = None,
    validate: bool = True,
    verbose: bool = True,
) -> BatchResult:
    """
    Run batch inference on a large dataset, streaming in chunks.

    Parameters
    ----------
    result : Result
        Fitted KiteML result.
    data : str or pd.DataFrame
        Input data — file path (CSV/Parquet) or DataFrame.
    chunk_size : int
        Number of rows per inference chunk. Default 1000.
    output_path : str, optional
        If provided, save predictions CSV to this path.
    validate : bool
        Run guardrail validation on each chunk.
    verbose : bool
        Print progress.

    Returns
    -------
    BatchResult
    """
    from kiteml.deployment.inference_guardrails import InferenceGuardrails

    t0 = time.perf_counter()
    guardrails = InferenceGuardrails(result.feature_names or []) if validate else None
    all_preds: list[np.ndarray] = []
    all_probas: list[np.ndarray] = []
    n_chunks = 0
    n_rows = 0
    has_proba = result.problem_type == "classification" and hasattr(result.model, "predict_proba")

    # ── Load data chunks ──────────────────────────────────────────────────
    def _chunk_dataframe(df: pd.DataFrame):
        for start in range(0, len(df), chunk_size):
            yield df.iloc[start : start + chunk_size]

    def _get_chunks():
        if isinstance(data, str):
            if data.endswith(".parquet"):
                df = pd.read_parquet(data)
                yield from _chunk_dataframe(df)
            else:
                yield from pd.read_csv(data, chunksize=chunk_size)
        elif isinstance(data, pd.DataFrame):
            yield from _chunk_dataframe(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    # ── Inference loop ────────────────────────────────────────────────────
    for chunk in _get_chunks():
        n_chunks += 1
        chunk_rows = len(chunk)
        n_rows += chunk_rows

        # Drop target if present
        if result.feature_names:
            present = [f for f in result.feature_names if f in chunk.columns]
            chunk = chunk[present]

        # Validate
        if guardrails is not None:
            gr = guardrails.validate(chunk)
            gr.raise_if_invalid()

        # Preprocess
        X = result.preprocessor.transform(chunk) if result.preprocessor is not None else chunk.values

        # Predict
        preds = result.model.predict(X)
        all_preds.append(preds)

        if has_proba:
            probas = result.model.predict_proba(X)
            all_probas.append(probas)

        if verbose:
            elapsed = time.perf_counter() - t0
            rps = n_rows / elapsed if elapsed > 0 else 0
            print(
                f"\r  Chunk {n_chunks:4d} | Rows processed: {n_rows:7,} | " f"{rps:,.0f} rows/s",
                end="",
                flush=True,
            )

    if verbose:
        print()  # newline after progress

    elapsed = time.perf_counter() - t0
    throughput = n_rows / elapsed if elapsed > 0 else 0.0
    predictions = np.concatenate(all_preds)
    probabilities = np.concatenate(all_probas) if all_probas else None

    batch_result = BatchResult(
        predictions=predictions,
        probabilities=probabilities,
        n_rows=n_rows,
        n_chunks=n_chunks,
        elapsed_s=round(elapsed, 3),
        throughput_rps=round(throughput, 1),
    )

    if verbose:
        print(f"✅ Batch complete: {n_rows:,} rows in {elapsed:.2f}s " f"({throughput:,.0f} rows/s)")

    if output_path:
        batch_result.save_csv(output_path)

    return batch_result
