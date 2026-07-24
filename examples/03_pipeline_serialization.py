#!/usr/bin/env python3
"""
03_pipeline_serialization.py — KiteML Pipeline Serialization Example

Demonstrates building a unified DAG pipeline, saving as .kml package with SHA-256 integrity,
and reloading in production.
"""

import pandas as pd
from kiteml import KiteMLPipeline


def main():
    print("=== KiteML Pipeline Serialization Example ===")

    df = pd.DataFrame({
        "age": [25, 45, 35, 50, 23, 62, 48, 29],
        "city": ["NYC", "LA", "NYC", "CHI", "LA", "NYC", "CHI", "LA"],
        "income": [50000, 90000, 62000, 110000, 48000, 125000, 98000, 54000],
        "churn": [1, 0, 0, 0, 1, 0, 0, 1]
    })

    # 1. Initialize orchestrator
    pipeline = KiteMLPipeline()

    # 2. Build DAG pipeline
    build_result = pipeline.build(df, target="churn")
    print(build_result.report.summary())

    # 3. Save as .kml binary package with SHA-256 checksum
    kml_path = "pipeline_package.kml"
    pipeline.save(kml_path)
    print(f"\nSaved DAG pipeline to {kml_path}")

    # 4. Load pipeline package
    loaded_pipeline = KiteMLPipeline.load(kml_path)
    print("Successfully loaded .kml package with verified SHA-256 checksum!")


if __name__ == "__main__":
    main()
