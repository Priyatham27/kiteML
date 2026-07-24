#!/usr/bin/env python3
"""
02_house_price_regression.py — KiteML Regression Example

Demonstrates simple AutoML regression training and evaluation metrics.
"""

import pandas as pd
from kiteml import train


def main():
    print("=== KiteML Regression Example ===")

    # Create synthetic housing DataFrame
    df = pd.DataFrame({
        "bedrooms": [2, 3, 4, 3, 5, 2, 4, 3],
        "bathrooms": [1.0, 2.0, 2.5, 1.5, 3.5, 1.0, 3.0, 2.0],
        "sqft": [1100, 1800, 2400, 1500, 3500, 950, 2800, 1750],
        "price": [310000, 480000, 620000, 410000, 890000, 275000, 750000, 460000]
    })

    # Train regression model
    result = train(data=df, target="price", problem_type="regression")

    print("\n--- Summary ---")
    print(result.summary())

    print("\n--- Evaluation Metrics ---")
    print(result.metrics)


if __name__ == "__main__":
    main()
