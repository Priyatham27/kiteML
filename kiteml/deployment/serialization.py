"""
serialization.py — Multi-format model serialization for KiteML.

Supports: joblib (default), pickle, and JSON metadata.
ONNX export lives in onnx_export.py.
"""

import json
import os
import pickle
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class SerializationResult:
    """Result of a serialization operation."""
    format: str
    path: str
    size_bytes: int
    checksum: str
    serialized_at: str


def _md5(path: str) -> str:
    """Compute MD5 hex digest of a file."""
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def save_joblib(obj: Any, path: str) -> SerializationResult:
    """Save any object with joblib (preferred for sklearn objects)."""
    import joblib
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(obj, path, compress=3)
    size = os.path.getsize(path)
    return SerializationResult(
        format="joblib", path=path, size_bytes=size,
        checksum=_md5(path),
        serialized_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def load_joblib(path: str) -> Any:
    """Load a joblib-serialized object."""
    import joblib
    return joblib.load(path)


def save_pickle(obj: Any, path: str) -> SerializationResult:
    """Save any object with pickle."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    size = os.path.getsize(path)
    return SerializationResult(
        format="pickle", path=path, size_bytes=size,
        checksum=_md5(path),
        serialized_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def load_pickle(path: str) -> Any:
    """Load a pickle-serialized object."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(data: Dict, path: str, indent: int = 2) -> SerializationResult:
    """Save a dictionary as JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    size = os.path.getsize(path)
    return SerializationResult(
        format="json", path=path, size_bytes=size,
        checksum=_md5(path),
        serialized_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def load_json(path: str) -> Dict:
    """Load a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def serialize(obj: Any, path: str, format: str = "joblib") -> SerializationResult:
    """
    Unified serialization entry point.

    Parameters
    ----------
    obj : any
    path : str
    format : str
        ``'joblib'`` (default), ``'pickle'``, or ``'json'``.
    """
    fmt = format.lower()
    if fmt == "joblib":
        return save_joblib(obj, path)
    elif fmt == "pickle":
        return save_pickle(obj, path)
    elif fmt == "json":
        if not isinstance(obj, dict):
            raise TypeError("JSON format requires a dict object.")
        return save_json(obj, path)
    else:
        raise ValueError(f"Unknown format '{format}'. Use 'joblib', 'pickle', or 'json'.")
