#!/usr/bin/env python3
"""
Customer Churn Prediction Example
"""

import pandas as pd
from kiteml import train


def main():
    print("=== Customer Churn Prediction ===")
    df = pd.DataFrame({
        "age": [25, 45, 35, 50, 23],
        "income": [50000, 90000, 62000, 110000, 48000],
        "churn": [1, 0, 0, 0, 1]
    })
    result = train(df, target="churn", problem_type="classification")
    print(result.summary())


if __name__ == "__main__":
    main()
