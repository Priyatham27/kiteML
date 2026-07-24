#!/usr/bin/env python3
"""
01_quickstart_classification.py — KiteML Classification Example

Demonstrates simple AutoML classification training, diagnostic checks, and inference.
"""

import pandas as pd
from kiteml import train, load


def main():
    print("=== KiteML Classification Example ===")

    # Create synthetic classification DataFrame
    df = pd.DataFrame({
        "age": [25, 45, 35, 50, 23, 62, 48, 29, 38, 55],
        "income": [50000, 90000, 62000, 110000, 48000, 125000, 98000, 54000, 71000, 115000],
        "credit_score": [650, 720, 680, 800, 610, 750, 710, 640, 690, 780],
        "churn": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0]
    })

    # Train classification model
    result = train(data=df, target="churn", problem_type="classification")

    # Print summary & execution diagnostics
    print("\n--- Summary ---")
    print(result.summary())

    print("\n--- Diagnostics ---")
    print(result.diagnostics())

    # Make predictions on new record
    new_data = pd.DataFrame({
        "age": [30],
        "income": [58000],
        "credit_score": [660]
    })
    preds = result.predict(new_data)
    print(f"\nPrediction for new record: {preds}")

    # Save model artifact
    result.save_model("example_churn_model.pkl")
    print("\nSaved model to example_churn_model.pkl")


if __name__ == "__main__":
    main()
