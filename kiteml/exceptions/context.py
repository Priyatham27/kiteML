"""
context.py — ErrorContext model for KiteML exception metadata.
"""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ErrorContext:
    """
    Structured context for KiteML exceptions.

    Attributes
    ----------
    operation : str, optional
        The operation or workflow step being executed when the error occurred.
    dataset_name : str, optional
        Name or path of the dataset.
    target : str, optional
        Target column name.
    available_columns : list of str, optional
        List of available columns in the dataset.
    row_count : int, optional
        Number of rows in the dataset.
    column_count : int, optional
        Number of columns in the dataset.
    feature_name : str, optional
        Specific feature column related to the error.
    model_name : str, optional
        Name of the model being trained/evaluated.
    metadata : dict, optional
        Additional key-value metadata mapping.
    """

    operation: str | None = None
    dataset_name: str | None = None
    target: str | None = None
    available_columns: list[str] | None = None
    row_count: int | None = None
    column_count: int | None = None
    feature_name: str | None = None
    model_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert ErrorContext to a dictionary, omitting None fields."""
        data = asdict(self)
        result: dict[str, Any] = {}
        for k, v in data.items():
            if (v is not None and v != {}) or (k == "metadata" and v):
                result[k] = v
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ErrorContext":
        """Construct ErrorContext from a dictionary."""
        known_fields = {
            "operation",
            "dataset_name",
            "target",
            "available_columns",
            "row_count",
            "column_count",
            "feature_name",
            "model_name",
            "metadata",
        }
        ctx_args = {k: v for k, v in data.items() if k in known_fields}
        extra_meta = {k: v for k, v in data.items() if k not in known_fields}

        if extra_meta:
            existing_meta = ctx_args.get("metadata", {})
            if isinstance(existing_meta, dict):
                merged_meta = {**existing_meta, **extra_meta}
                ctx_args["metadata"] = merged_meta

        return cls(**ctx_args)
