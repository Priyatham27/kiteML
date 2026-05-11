"""
benchmarks/benchmark_runner.py — Standardized benchmarking system.
"""
import time
import pandas as pd
from typing import Dict, Any

from kiteml.core import train

def run_benchmark(dataset_path: str, target_col: str) -> Dict[str, Any]:
    """Run standardized training and inference benchmarking."""
    print(f"Benchmarking KiteML on {dataset_path}...")
    
    df = pd.read_csv(dataset_path)
    
    # Measure Training Speed
    t0 = time.perf_counter()
    res = train(df, target=target_col, verbose=False)
    t1 = time.perf_counter()
    train_time = t1 - t0
    
    # Measure Inference Speed (Batch)
    from kiteml.deployment.realtime_inference import RealtimeInferenceEngine
    res.package("bench_temp.kiteml")
    engine = RealtimeInferenceEngine.from_bundle("bench_temp.kiteml")
    
    records = df.head(1000).to_dict(orient="records")
    t0 = time.perf_counter()
    for r in records:
        engine.predict(r)
    t1 = time.perf_counter()
    inf_time = (t1 - t0) / len(records) * 1000  # ms per record
    
    import os
    import shutil
    shutil.rmtree("bench_temp.kiteml", ignore_errors=True)
    
    score = res.metrics.score if hasattr(res.metrics, "score") else 0.0
    
    return {
        "dataset": dataset_path,
        "winner_model": res.model_name,
        "training_time_sec": round(train_time, 2),
        "inference_latency_ms": round(inf_time, 2),
        "accuracy_score": round(score, 4)
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python benchmark_runner.py <csv_path> <target>")
        sys.exit(1)
    
    report = run_benchmark(sys.argv[1], sys.argv[2])
    print("\n--- Benchmark Report ---")
    for k, v in report.items():
        print(f"{k.ljust(20)} : {v}")
