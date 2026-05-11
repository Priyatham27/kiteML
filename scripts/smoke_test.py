#!/usr/bin/env python3
"""
smoke_test.py — KiteML Smoke Test Suite

Quick validation that all critical KiteML features work end-to-end.
Generates synthetic data and runs through core workflows.

Usage:
  python scripts/smoke_test.py
"""

import sys
import tempfile
import traceback
from pathlib import Path


def check(label: str, func) -> bool:
    """Run a check function and report result."""
    try:
        func()
        print(f"  [PASS] {label}")
        return True
    except Exception:
        print(f"  [FAIL] {label}")
        traceback.print_exc()
        return False


def test_import():
    """Test basic import."""
    import kiteml
    assert kiteml.__version__


def test_train_classification():
    """Test classification training with synthetic data."""
    import pandas as pd
    from sklearn.datasets import make_classification

    from kiteml import train

    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f.name, index=False)
        result = train(f.name, target="target")

    assert result is not None
    result.summary()


def test_train_regression():
    """Test regression training with synthetic data."""
    import pandas as pd
    from sklearn.datasets import make_regression

    from kiteml import train

    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f.name, index=False)
        result = train(f.name, target="target")

    assert result is not None


def test_model_save_load():
    """Test model save and load."""
    import pandas as pd
    from sklearn.datasets import make_classification

    from kiteml import train

    X, y = make_classification(n_samples=80, n_features=4, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f.name, index=False)
        result = train(f.name, target="target")

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        result.save_model(f.name)
        assert Path(f.name).stat().st_size > 0


def test_cli_parser():
    """Test CLI parser builds without errors."""
    from kiteml.cli.parser import build_parser

    parser = build_parser()
    assert parser is not None
    assert parser.prog == "kiteml"


def test_result_class():
    """Test Result class instantiation."""
    from kiteml.output.result import Result
    assert Result is not None


def main() -> int:
    print("\n" + "=" * 50)
    print("  KiteML Smoke Tests")
    print("=" * 50 + "\n")

    tests = [
        ("Import kiteml", test_import),
        ("Classification training", test_train_classification),
        ("Regression training", test_train_regression),
        ("Model save/load", test_model_save_load),
        ("CLI parser builds", test_cli_parser),
        ("Result class", test_result_class),
    ]

    results = []
    for label, func in tests:
        results.append(check(label, func))

    passed = sum(results)
    total = len(results)
    failed = total - passed

    print(f"\n{'=' * 50}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print("=" * 50)

    if failed == 0:
        print("\n  All smoke tests PASSED.\n")
        return 0
    else:
        print(f"\n  {failed} smoke test(s) FAILED.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
