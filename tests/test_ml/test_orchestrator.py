"""
test_orchestrator.py — Unit tests for train() and load() (Story 5.8).
"""

import pandas as pd

from kiteml import train


def test_train_and_load_workflow(tmp_path):
    # Use enough rows for 5-fold stratified CV (need >=5 samples per class)
    df = pd.DataFrame(
        {
            "age": [
                20,
                25,
                30,
                35,
                40,
                45,
                50,
                55,
                22,
                27,
                32,
                37,
                42,
                47,
                52,
                57,
                23,
                28,
                33,
                38,
            ],
            "salary": [
                30000,
                40000,
                50000,
                60000,
                70000,
                80000,
                90000,
                100000,
                32000,
                42000,
                52000,
                62000,
                72000,
                82000,
                92000,
                102000,
                31000,
                41000,
                51000,
                61000,
            ],
            "purchased": [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1],
        }
    )

    result = train(dataframe=df, target="purchased", validate_data=False)
    assert result is not None
    assert result.model is not None
    assert result.model_name is not None
    assert result.problem_type == "classification"
    result.summary()  # should not raise

    pkg_path = str(tmp_path / "model.pkl")
    result.save(pkg_path)

    from kiteml.output.result import Result

    bundle = Result.load(pkg_path)
    # bundle is a dict with 'model', 'preprocessor', etc.
    assert "model" in bundle
    assert "preprocessor" in bundle
