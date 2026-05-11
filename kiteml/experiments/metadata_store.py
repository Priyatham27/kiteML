"""
metadata_store.py — Persistent key-value metadata store for KiteML experiments.

Provides a simple JSON-backed store for arbitrary metadata associated
with models, experiments, and runs.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

_DEFAULT_STORE = os.path.join(os.path.expanduser("~"), ".kiteml", "metadata")


class MetadataStore:
    """
    Simple JSON-backed metadata store.

    Parameters
    ----------
    namespace : str
        Logical namespace (e.g. model name or experiment name).
    store_path : str, optional
        Override default store root directory.
    """

    def __init__(self, namespace: str = "default", store_path: Optional[str] = None):
        self.namespace = namespace
        self._root = store_path or _DEFAULT_STORE
        self._path = os.path.join(self._root, f"{namespace}.json")
        os.makedirs(self._root, exist_ok=True)
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self._path):
            with open(self._path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, default=str)

    def set(self, key: str, value: Any) -> None:
        """Store a metadata value."""
        self._data[key] = {
            "value": value,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._save()

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a metadata value."""
        entry = self._data.get(key)
        if entry is None:
            return default
        return entry.get("value", default)

    def delete(self, key: str) -> bool:
        """Delete a metadata key. Returns True if it existed."""
        if key in self._data:
            del self._data[key]
            self._save()
            return True
        return False

    def keys(self) -> List[str]:
        """Return all stored keys."""
        return list(self._data.keys())

    def all(self) -> Dict[str, Any]:
        """Return all stored metadata as plain dict."""
        return {k: v.get("value") for k, v in self._data.items()}

    def clear(self) -> None:
        """Clear all metadata in this namespace."""
        self._data = {}
        self._save()

    def __repr__(self) -> str:
        return f"<MetadataStore namespace='{self.namespace}' keys={len(self._data)}>"
