"""
commands/benchmark.py — CLI command for benchmarking models.
"""

import os
import time

from kiteml.cli.ui.colors import print_error, print_header, print_info, print_step, print_success


def setup_benchmark_parser(subparsers):
    parser = subparsers.add_parser("benchmark", aliases=["bench"], help="Benchmark KiteML model performance")
    parser.add_argument("model", type=str, help="Path to .kiteml bundle")
    parser.add_argument("data", type=str, help="Path to inference data")
    parser.set_defaults(func=run_benchmark)


def run_benchmark(args):
    if not os.path.exists(args.model) or not os.path.exists(args.data):
        print_error("Model or data not found.")
        return 1

    print_header("KiteML Benchmarking Suite")
    print_info(f"Loading bundle: {args.model}")
    import pandas as pd

    from kiteml.deployment.realtime_inference import RealtimeInferenceEngine

    engine = RealtimeInferenceEngine.from_bundle(args.model)
    df = pd.read_csv(args.data) if args.data.endswith(".csv") else pd.read_parquet(args.data)
    records = df.to_dict(orient="records")

    # Warmup
    for r in records[:5]:
        engine.predict(r)

    # Benchmark
    print_info(f"Running inference on {len(records)} records...")
    t0 = time.perf_counter()
    for r in records:
        engine.predict(r)
    t1 = time.perf_counter()

    total_time = t1 - t0
    throughput = len(records) / total_time
    latency_ms = (total_time / len(records)) * 1000

    print_step(f"Total Time : {total_time:.4f}s")
    print_step(f"Throughput : {throughput:.1f} req/sec")
    print_step(f"Avg Latency: {latency_ms:.2f} ms/req")

    print_success("Benchmark complete.")
    return 0
