#!/usr/bin/env python3
"""
Titanic Survival Prediction Example
"""

import pandas as pd
from kiteml import train


def main():
    print("=== Titanic Survival Prediction ===")
    df = pd.DataFrame({
        "pclass": [1, 3, 3, 1, 3],
        "sex": ["female", "male", "female", "female", "male"],
        "age": [29, 24, 26, 35, 54],
        "survived": [1, 0, 1, 1, 0]
    })
    result = train(df, target="survived", problem_type="classification")
    print(result.summary())


if __name__ == "__main__":
    main()
