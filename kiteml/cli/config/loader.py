"""
config/loader.py — Config file loader for KiteML workflows.
"""

import os

import yaml


def load_config(path: str) -> dict:
    """Load a KiteML workflow configuration file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            return yaml.safe_load(f)
        elif path.endswith(".json"):
            import json

            return json.load(f)

    raise ValueError("Config file must be .yaml, .yml, or .json")
