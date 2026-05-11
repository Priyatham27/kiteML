"""
onnx_export.py — Export KiteML models to ONNX format (optional dependency).

ONNX enables cross-platform, cross-language model serving without needing
Python or scikit-learn in the production environment.

Requires: pip install skl2onnx onnx
"""

import os
from dataclasses import dataclass
from typing import Any, List


@dataclass
class OnnxExportResult:
    """Result of an ONNX export operation."""

    path: str
    size_bytes: int
    opset_version: int
    model_type: str
    n_features: int
    note: str = ""


def export_onnx(
    model: Any,
    feature_names: List[str],
    path: str = "model.onnx",
    initial_type_hint: str = "float32",
    opset_version: int = 17,
) -> OnnxExportResult:
    """
    Export a fitted sklearn estimator to ONNX format.

    Parameters
    ----------
    model : fitted sklearn estimator
    feature_names : list of str
    path : str
        Output file path. Default ``'model.onnx'``.
    initial_type_hint : str
        Input dtype — ``'float32'`` (default) or ``'double'``.
    opset_version : int
        ONNX opset version. Default 17.

    Returns
    -------
    OnnxExportResult

    Raises
    ------
    ImportError
        If ``skl2onnx`` or ``onnx`` are not installed.
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import DoubleTensorType, FloatTensorType
    except ImportError:
        raise ImportError("ONNX export requires skl2onnx and onnx.\n" "Install with: pip install skl2onnx onnx")

    n_features = len(feature_names)
    dtype = FloatTensorType if initial_type_hint == "float32" else DoubleTensorType
    initial_type = [("float_input", dtype([None, n_features]))]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=opset_version,
    )

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    size = os.path.getsize(path)
    print(f"✅ ONNX model saved → {path}  ({size/1024:.1f} KB)")

    return OnnxExportResult(
        path=os.path.abspath(path),
        size_bytes=size,
        opset_version=opset_version,
        model_type=type(model).__name__,
        n_features=n_features,
    )


def validate_onnx(path: str, X_sample: Any) -> bool:
    """
    Validate an ONNX model by running a sample inference.

    Parameters
    ----------
    path : str
    X_sample : array-like, shape (n, n_features)

    Returns
    -------
    bool
        True if inference runs without error.
    """
    try:
        import numpy as np
        import onnxruntime as rt

        sess = rt.InferenceSession(path)
        input_name = sess.get_inputs()[0].name
        X = np.array(X_sample, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        sess.run(None, {input_name: X})
        return True
    except ImportError:
        print("⚠️  onnxruntime not installed. Skipping validation.")
        return False
    except Exception as e:
        print(f"❌ ONNX validation failed: {e}")
        return False
