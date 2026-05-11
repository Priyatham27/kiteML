"""
commands/predict.py — CLI command for batch inference.
"""
import argparse
import os

from kiteml.cli.ui.colors import print_step, print_info, print_error, print_success
from kiteml.deployment.packaging import load_bundle
import pandas as pd


def setup_predict_parser(subparsers):
    parser = subparsers.add_parser("predict", aliases=["p"], help="Run batch predictions using a trained model bundle")
    parser.add_argument("model", type=str, help="Path to the .kiteml bundle")
    parser.add_argument("data", type=str, help="Path to input data (CSV/Parquet)")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output path for predictions")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Rows to process per batch")
    parser.set_defaults(func=run_predict)


def run_predict(args):
    if not os.path.exists(args.model):
        print_error(f"Model bundle not found: {args.model}")
        return 1
    if not os.path.exists(args.data):
        print_error(f"Input data not found: {args.data}")
        return 1

    print_info(f"Loading bundle: {args.model}")
    from kiteml.deployment.batch_inference import BatchInferenceEngine
    
    try:
        engine = BatchInferenceEngine.from_bundle(args.model)
        print_info(f"Running predictions on {args.data} (chunk size: {args.chunk_size})")
        res = engine.predict_file(args.data, chunk_size=args.chunk_size, output_path=args.output)
    except Exception as e:
        print_error(f"Batch prediction failed: {e}")
        return 1

    print_step(f"Processed {res.n_rows} rows in {res.n_chunks} chunks.")
    print_success(f"Predictions saved to {args.output}")
    return 0
