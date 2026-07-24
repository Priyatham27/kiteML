#!/usr/bin/env python3
"""
House Prices Regression Example
"""

import pandas as pd
from kiteml import train


def main():
    print("=== House Prices Regression ===")
    df = pd.DataFrame({
        "bedrooms": [2, 3, 4, 3, 5],
        "sqft": [1100, 1800, 2400, 1500, 3500],
        "price": [310000, 480000, 620000, 410000, 890000]
    })
    result = train(df, target="price", problem_type="regression")
    print(result.summary())


if __name__ == "__main__":
    main()
