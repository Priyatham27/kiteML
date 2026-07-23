"""
engine.py — Core validation engine for executing KiteML validation pipelines.
"""

import time
from typing import Any

import pandas as pd

from kiteml.validation.utils import get_dataframe_memory_mb
from kiteml.validation.validation_report import ValidationReport
from kiteml.validation.validator import BaseValidator


class ValidationEngine:
    """
    Orchestrates execution of multiple validators against a dataset.

    Hierarchy:
    ValidationEngine
      ├── DatasetValidator (Story 2.2)
      ├── TargetValidator  (Story 2.3)
      ├── SchemaValidator  (Story 2.4)
      └── QualityValidator (Story 2.5)
    """

    def __init__(self, validators: list[BaseValidator] | None = None) -> None:
        self.validators: list[BaseValidator] = list(validators) if validators else []

    def add_validator(self, validator: BaseValidator) -> "ValidationEngine":
        """
        Add a validator to the validation pipeline.

        Parameters
        ----------
        validator : BaseValidator

        Returns
        -------
        ValidationEngine (self for method chaining)
        """
        self.validators.append(validator)
        return self

    def remove_validator(self, name: str) -> bool:
        """
        Remove a validator by name.

        Returns
        -------
        bool
            True if a validator was removed.
        """
        initial_len = len(self.validators)
        self.validators = [v for v in self.validators if v.name != name]
        return len(self.validators) < initial_len

    def clear(self) -> None:
        """Remove all validators from the engine."""
        self.validators.clear()

    def validate(
        self,
        df: pd.DataFrame,
        stop_on_error: bool = False,
        **kwargs: Any,
    ) -> ValidationReport:
        """
        Run all registered validators against the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to validate.
        stop_on_error : bool, default=False
            If True, halt execution of subsequent validators when an error is encountered.
        **kwargs : Any
            Additional contextual parameters passed to validators (e.g. target_column).

        Returns
        -------
        ValidationReport
        """
        start_time = time.perf_counter()
        report = ValidationReport()

        # Collect dataset metadata
        if isinstance(df, pd.DataFrame):
            report.dataset_metadata = {
                "n_rows": int(len(df)),
                "n_cols": int(len(df.columns)),
                "memory_mb": round(get_dataframe_memory_mb(df), 2),
            }

        for validator in self.validators:
            res = validator.validate(df, **kwargs)
            report.add_result(res)

            if stop_on_error and res.has_errors():
                break

        report.total_execution_time = round(time.perf_counter() - start_time, 6)
        return report

    def __len__(self) -> int:
        return len(self.validators)

    def __repr__(self) -> str:
        return f"<ValidationEngine(validators={len(self.validators)})>"
