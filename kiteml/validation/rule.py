"""
rule.py — Abstract base class for KiteML validation rules.
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from kiteml.validation.message import ValidationMessage
from kiteml.validation.severity import ValidationSeverity


class ValidationRule(ABC):
    """
    Abstract Base Class for an individual validation rule.

    Each rule performs a single, modular validation check on a dataset or metadata,
    and returns zero, one, or multiple `ValidationMessage` instances.
    """

    rule_id: str = "R000"
    name: str = "Base Validation Rule"
    description: str = "Base description for validation rule."
    default_severity: ValidationSeverity = ValidationSeverity.ERROR

    @abstractmethod
    def check(
        self,
        df: pd.DataFrame,
        **kwargs: Any,
    ) -> ValidationMessage | list[ValidationMessage] | None:
        """
        Execute the rule against a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to validate.
        **kwargs : Any
            Additional contextual inputs (e.g. target_column, problem_type).

        Returns
        -------
        ValidationMessage | list[ValidationMessage] | None
            One or more validation messages if issues are detected, or None if passed.
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id='{self.rule_id}', name='{self.name}')>"
