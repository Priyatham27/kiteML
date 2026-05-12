"""
data_contracts.py — Formal data contracts for production input validation.

Defines the expected schema for production inputs, with type, range, and
nullability constraints.  Used by guardrails for strict enforcement.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class FeatureContract:
    """Contract for a single feature."""

    name: str
    dtype: str  # "float", "int", "str", "bool"
    nullable: bool = False
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[Any] | None = None  # for categoricals
    description: str = ""


@dataclass
class DataContract:
    """Full data contract for a model's inputs."""

    model_name: str
    version: str
    features: dict[str, FeatureContract]
    created_at: str = ""

    def validate(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate a single row dict against the contract. Returns (ok, errors)."""
        errors: list[str] = []
        for feat_name, contract in self.features.items():
            if feat_name not in data:
                if not contract.nullable:
                    errors.append(f"Missing required field: '{feat_name}'")
                continue

            val = data[feat_name]

            if val is None:
                if not contract.nullable:
                    errors.append(f"'{feat_name}' cannot be null.")
                continue

            # Range check
            if contract.min_value is not None:
                try:
                    if float(val) < contract.min_value:
                        errors.append(f"'{feat_name}'={val} < min({contract.min_value})")
                except (TypeError, ValueError):
                    pass

            if contract.max_value is not None:
                try:
                    if float(val) > contract.max_value:
                        errors.append(f"'{feat_name}'={val} > max({contract.max_value})")
                except (TypeError, ValueError):
                    pass

            # Allowed values
            if contract.allowed_values is not None and val not in contract.allowed_values:
                errors.append(f"'{feat_name}'={val!r} not in allowed: {contract.allowed_values}")

        return len(errors) == 0, errors

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "created_at": self.created_at,
            "features": {name: fc.__dict__ for name, fc in self.features.items()},
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_result(cls, result: Any, version: str = "1.0.0") -> "DataContract":
        """Build a DataContract from a KiteML Result + DataProfile."""
        import time

        features: dict[str, FeatureContract] = {}

        for feat in result.feature_names or []:
            dtype = "float"
            nullable = False

            # Enrich from profile if available
            if result.data_profile is not None:
                try:
                    col_type = result.data_profile.column_analysis.profiles.get(feat)
                    if col_type:
                        ct = col_type.column_type.value
                        if ct == "categorical":
                            dtype = "str"
                        elif ct == "boolean":
                            dtype = "bool"
                        nullable = col_type.null_ratio > 0
                except Exception:
                    pass

            features[feat] = FeatureContract(name=feat, dtype=dtype, nullable=nullable)

        return cls(
            model_name=result.model_name,
            version=version,
            features=features,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
