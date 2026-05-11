"""
signatures.py — Model fingerprinting and integrity verification for KiteML.

Generates a cryptographic fingerprint of a trained model and its configuration
so that any tampering or unexpected changes can be detected.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelSignature:
    """Cryptographic fingerprint of a fitted model."""

    model_name: str
    fingerprint: str  # MD5 hex digest of model bytes
    feature_hash: str  # MD5 of sorted feature names
    config_hash: str  # MD5 of config dict
    created_at: str
    n_features: int

    def verify(self, other: "ModelSignature") -> bool:
        """Check if two signatures match (identity check)."""
        return self.fingerprint == other.fingerprint and self.feature_hash == other.feature_hash

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def sign_model(result: Any) -> ModelSignature:
    """
    Generate a cryptographic fingerprint for a KiteML Result.

    Parameters
    ----------
    result : Result

    Returns
    -------
    ModelSignature
    """
    import tempfile

    import joblib

    # Fingerprint model bytes
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        joblib.dump(result.model, tmp_path, compress=0)
        h = hashlib.md5()
        with open(tmp_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        fingerprint = h.hexdigest()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Feature hash
    features = sorted(result.feature_names or [])
    feature_hash = hashlib.md5(json.dumps(features).encode()).hexdigest()

    # Config hash
    from kiteml import config as cfg

    config_dict = {
        k: v for k, v in cfg.__dict__.items() if not k.startswith("_") and isinstance(v, (int, float, str, bool))
    }
    config_hash = hashlib.md5(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()

    return ModelSignature(
        model_name=result.model_name,
        fingerprint=fingerprint,
        feature_hash=feature_hash,
        config_hash=config_hash,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        n_features=len(result.feature_names or []),
    )
