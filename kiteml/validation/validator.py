"""
validator.py — Abstract Base Class for KiteML validators.
"""

import time
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from kiteml.validation.rule import ValidationRule
from kiteml.validation.utils import get_dataframe_memory_mb
from kiteml.validation.validation_result import ValidationResult


class BaseValidator(ABC):
    """
    Abstract Base Class for composite validators.

    A validator aggregates a set of `ValidationRule` instances and executes them
    sequentially against a dataset to produce a `ValidationResult`.
    """

    description: str = "Base validator class."
    rules: list[ValidationRule] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the validator."""
        pass

    def __init__(self, rules: list[ValidationRule] | None = None) -> None:
        if rules is not None:
            self.rules = list(rules)

    def validate(
        self,
        df: pd.DataFrame,
        **kwargs: Any,
    ) -> ValidationResult:
        """
        Run validation rules against the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to validate.
        **kwargs : Any
            Additional options passed to rule execution (e.g. target_column).

        Returns
        -------
        ValidationResult
        """
        start_time = time.perf_counter()
        result = ValidationResult(validator_name=self.name)

        # Collect baseline statistics if df is a DataFrame
        if isinstance(df, pd.DataFrame):
            result.statistics["n_rows"] = int(len(df))
            result.statistics["n_cols"] = int(len(df.columns))
            result.statistics["memory_mb"] = round(get_dataframe_memory_mb(df), 2)

        # Execute registered rules
        for rule in self.rules:
            try:
                msgs = rule.check(df, **kwargs)
                if msgs is None:
                    continue
                if isinstance(msgs, list):
                    for msg in msgs:
                        result.add_message(msg)
                else:
                    result.add_message(msgs)
            except Exception as exc:
                result.add_error(
                    title=f"Rule Execution Failure ({rule.rule_id})",
                    description=f"Rule {rule.name} raised unexpected error: {exc}",
                    suggestion="Verify input data structure or rule implementation.",
                    rule_id=rule.rule_id,
                    code="KML_RULE_ERROR",
                    context={"exception": str(exc)},
                )

        result.execution_time = round(time.perf_counter() - start_time, 6)
        return result

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', rules={len(self.rules)})>"
