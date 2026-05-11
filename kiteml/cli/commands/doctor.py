"""
commands/doctor.py — Diagnostics and environment check for KiteML.
"""

import sys

from kiteml.cli.ui.colors import print_error, print_header, print_info, print_step, print_warning


def setup_doctor_parser(subparsers):
    parser = subparsers.add_parser("doctor", help="Diagnose KiteML environment and dependencies")
    parser.set_defaults(func=run_doctor)


def run_doctor(args):
    print_header("KiteML Diagnostics")

    # 1. Python version
    py_version = sys.version.split()[0]
    if sys.version_info >= (3, 8):
        print_step(f"Python version: {py_version} (Compatible)")
    else:
        print_warning(f"Python version: {py_version} (Warning: 3.8+ recommended)")

    # 2. Core Dependencies
    try:
        import numpy
        import pandas
        import sklearn

        print_step(f"scikit-learn : {sklearn.__version__}")
        print_step(f"pandas       : {pandas.__version__}")
        print_step(f"numpy        : {numpy.__version__}")
    except ImportError as e:
        print_error(f"Missing core dependency: {e}")

    # 3. Optional Dependencies
    print_header("Optional Features")
    try:
        import fastapi
        import uvicorn

        print_step(f"FastAPI / Serving : Installed (fastapi {fastapi.__version__})")
    except ImportError:
        print_warning("FastAPI / Serving : Missing (run `pip install fastapi uvicorn`)")

    try:
        import onnx
        import skl2onnx

        print_step("ONNX Export       : Installed")
    except ImportError:
        print_warning("ONNX Export       : Missing (run `pip install onnx skl2onnx`)")

    try:
        import xgboost

        print_step("XGBoost           : Installed")
    except ImportError:
        print_warning("XGBoost           : Missing (run `pip install xgboost`)")

    try:
        import lightgbm

        print_step("LightGBM          : Installed")
    except ImportError:
        print_warning("LightGBM          : Missing (run `pip install lightgbm`)")

    print_info("\nYour KiteML environment is ready.")
    return 0
