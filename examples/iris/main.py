#!/usr/bin/env python3
"""
Iris Multiclass Classification Example
"""

import pandas as pd
from kiteml import train


def main():
    print("=== Iris Multiclass Classification ===")
    df = pd.DataFrame({
        "sepal_length": [5.1, 4.9, 6.2, 5.9, 6.5],
        "sepal_width": [3.5, 3.0, 2.8, 3.0, 3.0],
        "species": ["setosa", "setosa", "versicolor", "versicolor", "virginica"]
    })
    result = train(df, target="species", problem_type="classification")
    print(result.summary())


if __name__ == "__main__":
    main()
